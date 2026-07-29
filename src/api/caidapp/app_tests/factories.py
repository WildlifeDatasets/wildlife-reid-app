import factory
from django.contrib.auth import get_user_model
from django.utils import timezone

from caidapp import models

User = get_user_model()


class UserFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = User

    username = factory.Sequence(lambda n: f"user{n}")
    email = factory.LazyAttribute(lambda o: f"{o.username}@example.com")

    @factory.post_generation
    def password(obj, create, extracted, **kwargs):
        password = extracted or "test123"
        obj.set_password(password)
        if create:
            obj.save()


class WorkGroupFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = models.WorkGroup

    name = factory.Sequence(lambda n: f"workgroup{n}")


class CaidUserFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = models.CaIDUser
        django_get_or_create = ("user",)

    user = factory.SubFactory(UserFactory)
    workgroup = factory.SubFactory(WorkGroupFactory)
    workgroup_admin = False

    class Params:
        admin = factory.Trait(
            workgroup_admin=True
        )


class TaxonFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = models.Taxon

    name = factory.Sequence(lambda n: f"Taxon{n}")


class LocalityFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = models.Locality

    name = factory.Sequence(lambda n: f"Locality{n}")

    owner = factory.SubFactory(CaidUserFactory)


class UploadedArchiveFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = models.UploadedArchive

    name = factory.Sequence(lambda n: f"archive{n}")
    owner = factory.SubFactory(CaidUserFactory)
    uploaded_at = factory.LazyFunction(timezone.now)

    taxon_status = "TV"
    identification_status = "C"
    import_finished = True
    contains_single_taxon = False


class SequenceFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = models.Sequence

    uploaded_archive = factory.SubFactory(UploadedArchiveFactory)
    local_id = factory.Sequence(lambda n: n)


class IndividualIdentityFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = models.IndividualIdentity

    name = factory.Sequence(lambda n: f"Identity{n}")
    owner_workgroup = factory.SubFactory(WorkGroupFactory)



def _set_mediafile_observation_value(mediafile, create, field, value, ignored_values=(None,)):
    if not create or value in ignored_values:
        return
    observation = mediafile.observations.order_by("id").first()
    if observation is None:
        observation = models.AnimalObservation(mediafile=mediafile)
    setattr(observation, field, value)
    observation.save()


class MediaFileFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = models.MediaFile

    parent = factory.SubFactory(UploadedArchiveFactory)
    captured_at = factory.LazyFunction(timezone.now)
    mediafile = factory.django.ImageField(color="blue")
    image_file = factory.django.ImageField(color="blue")
    preview = factory.django.ImageField(color="blue")

    original_filename = factory.Sequence(lambda n: f"image{n}.jpg")

    media_type = "image"

    @factory.post_generation
    def identity(self, create, extracted, **kwargs):
        _set_mediafile_observation_value(self, create, "identity", extracted)

    @factory.post_generation
    def with_identity(self, create, extracted, **kwargs):
        if create and extracted:
            identity = IndividualIdentityFactory(owner_workgroup=self.parent.owner.workgroup)
            _set_mediafile_observation_value(self, create, "identity", identity)

    @factory.post_generation
    def representative(self, create, extracted, **kwargs):
        if create and extracted:
            observation = self.observations.order_by("id").first()
            if observation is None or observation.identity is None:
                identity = IndividualIdentityFactory(owner_workgroup=self.parent.owner.workgroup)
                _set_mediafile_observation_value(self, create, "identity", identity)
            _set_mediafile_observation_value(self, create, "identity_is_representative", True)

    @factory.post_generation
    def identity_is_representative(self, create, extracted, **kwargs):
        _set_mediafile_observation_value(
            self,
            create,
            "identity_is_representative",
            extracted,
            ignored_values=(None, False),
        )

    @factory.post_generation
    def taxon(self, create, extracted, **kwargs):
        _set_mediafile_observation_value(self, create, "taxon", extracted)

    @factory.post_generation
    def predicted_taxon(self, create, extracted, **kwargs):
        _set_mediafile_observation_value(self, create, "predicted_taxon", extracted)

    @factory.post_generation
    def predicted_taxon_confidence(self, create, extracted, **kwargs):
        _set_mediafile_observation_value(self, create, "predicted_taxon_confidence", extracted)

    @factory.post_generation
    def orientation(self, create, extracted, **kwargs):
        _set_mediafile_observation_value(self, create, "orientation", extracted, ignored_values=(None, "N"))

    @factory.post_generation
    def taxon_verified(self, create, extracted, **kwargs):
        _set_mediafile_observation_value(self, create, "taxon_verified", extracted, ignored_values=(None, False))

    @factory.post_generation
    def taxon_verified_at(self, create, extracted, **kwargs):
        _set_mediafile_observation_value(self, create, "taxon_verified_at", extracted)

    @factory.post_generation
    def animal_number(self, create, extracted, **kwargs):
        # There is no observation equivalent. Accept the old test keyword while
        # legacy tests are being migrated, but deliberately do not persist it.
        return


class AnimalObservationFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = models.AnimalObservation

    mediafile = factory.SubFactory(MediaFileFactory)
    taxon = factory.SubFactory(TaxonFactory)
    taxon_verified = True


class WorkGroupInvitationFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = models.WorkGroupInvitation

    invited_user = factory.SubFactory(CaidUserFactory)
    invited_by = factory.SubFactory(CaidUserFactory)
    target_workgroup = factory.LazyAttribute(lambda o: o.invited_user.workgroup)
    status = "pending"


class AlbumFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = models.Album

    name = factory.Sequence(lambda n: f"album{n}")
    owner = factory.SubFactory(CaidUserFactory)



# class NotificationFactory(factory.django.DjangoModelFactory):
#     class Meta:
#         model = models.Notification
#
#     recipients = factory.RelatedFactoryList(
#         "caidapp.models.NotificationRecipient",
#         factory_related_name="notification",
#         size=1,
#     )
#     message = factory.Sequence(lambda n: f"Notification message {n}")
#     read = False

class NotificationFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = models.Notification

    message = factory.Sequence(lambda n: f"Notification message {n}")
    level = models.Notification.INFO


class NotificationRecipientFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = models.NotificationRecipient

    notification = factory.SubFactory(NotificationFactory)
    user = factory.SubFactory(CaidUserFactory)

    read = False
