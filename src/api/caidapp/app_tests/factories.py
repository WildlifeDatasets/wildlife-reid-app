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
    mediafiles_imported = True
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
    orientation = "N"

    identity = None

    class Params:

        with_identity = factory.Trait(
            identity=factory.SubFactory(IndividualIdentityFactory)
        )

        representative = factory.Trait(
            identity_is_representative=True,
            identity=factory.SubFactory(IndividualIdentityFactory)
        )


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