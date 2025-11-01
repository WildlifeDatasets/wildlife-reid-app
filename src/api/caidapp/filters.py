import django_filters
from django.db.models import Value
from django.db.models.functions import Concat

from . import models


class LocalityFilter(django_filters.FilterSet):
    search = django_filters.CharFilter(method="filter_search", label="Search")
    area = django_filters.ModelChoiceFilter(queryset=models.Area.objects.all().order_by("name"))

    class Meta:
        model = models.Locality
        fields = {
            "name": ["icontains"],
            # "area": ['exact'],
        }

    def filter_search(self, queryset, name, value):
        # Annotate the queryset with a computed 'search' field.
        queryset = queryset.annotate(
            search=Concat(
                "name",
                Value(" "),
                "area__name",
            )
        )
        # Now filter on the annotated 'search' field.
        return queryset.filter(search__icontains=value)


class IndividualIdentityFilter(django_filters.FilterSet):
    mediafile_count = django_filters.RangeFilter(label="Count of Media Files")

    last_seen = django_filters.DateFromToRangeFilter(
        label="Last Seen",
        help_text="Enter dates in YYYY-MM-DD format",
        widget=django_filters.widgets.RangeWidget(attrs={"type": "date"}),
    )
    search = django_filters.CharFilter(method="filter_search", label="Search")

    class Meta:
        model = models.IndividualIdentity
        fields = {
            "name": ["icontains"],
            "code": ["icontains"],
            "juv_code": ["icontains"],
            "sex": ["exact"],
            "coat_type": ["exact"],
        }

    def filter_search(self, queryset, name, value):
        # Annotate the queryset with a computed 'search' field.
        queryset = queryset.annotate(
            search=Concat(
                "name",
                Value(" "),
                "code",
                Value(" "),
                "juv_code",
            )
        )
        # Now filter on the annotated 'search' field.
        return queryset.filter(search__icontains=value)


import django_filters

from .models import Taxon, UploadedArchive


class MediaFileFilter(django_filters.FilterSet):
    # A free-text search filter. This uses a custom method to apply full text search.
    # query = django_filters.CharFilter(method='filter_query', label='Search')

    # Custom filters for your parameters. The GET parameter names should match these.
    # album_hash = django_filters.CharFilter(method='filter_album', label='Album')
    # individual_identity_id = django_filters.NumberFilter(method='filter_identity', label='Identity')
    # taxon_id = django_filters.NumberFilter(method='filter_taxon', label='Taxon')
    # uploadedarchive_id = django_filters.NumberFilter(method='filter_uploaded_archive', label='Archive')
    # identity_is_representative = django_filters.BooleanFilter(field_name='identity_is_representative')
    # locality_hash = django_filters.CharFilter(method='filter_locality', label='Locality')
    # search = django_filters.CharFilter(label='Search')
    request = None
    taxon = django_filters.ModelChoiceFilter(queryset=models.Taxon.objects.all().order_by("name"))
    uploadedarchive = django_filters.ModelChoiceFilter(
        queryset=models.UploadedArchive.objects.none()
        # .annotate(
        #     name_extended=Concat(
        #         'name',
        #         Value(' ('),
        #         Cast('uploaded_at', output_field=CharField()),
        #         Value(')')
        #     )
        # )
        .order_by("-uploaded_at"),
        # annotate(
        # name_extended=Concat('name', Value(' ( asd'),  Value(')'))
        # ).order_by('-uploaded_at'),
        # label_from_instance=lambda obj: f"{obj.name} ({obj.uploaded_at.strftime('%Y-%m-%d')})",
        label="Uploaded Archive",
        # label_from_instance=lambda obj: obj.name_extended
    )
    #
    captured_at = django_filters.DateFromToRangeFilter(
        label="Captured At",
        help_text="Enter dates in YYYY-MM-DD format",
        widget=django_filters.widgets.RangeWidget(attrs={"type": "date"}),
    )
    search = django_filters.CharFilter(method="filter_search", label="Search")

    class Meta:
        model = models.MediaFile
        # Declare the fields you want to filter by.
        taxon = Taxon.objects.all().order_by("name")
        fields = {
            "locality__name": ["icontains"],
            "identity__name": ["icontains"],
            "media_type": ["exact"],
            # "taxon" : ["exact"],
            "orientation": ["exact"],
            "identity_is_representative": ["exact"],
            # "search": ["icontains"],
            # "taxon_verified": ["exact"],
        }

    def filter_search(self, queryset, name, value):
        # Annotate the queryset with a computed 'search' field.
        queryset = queryset.annotate(
            search=Concat(
                "taxon__name",
                Value(" "),
                "locality__name",
                Value(" "),
                "identity__name",
            )
        )
        # Now filter on the annotated 'search' field.
        return queryset.filter(search__icontains=value)

    def __init__(self, *args, **kwargs):
        self.request = kwargs.pop("request", None)
        if self.request is None:
            raise ValueError("request must be provided to MediaFileFilter")
        caiduser = self.request.user.caiduser
        super().__init__(*args, **kwargs)

        from .model_extra import user_has_access_to_uploadedarchives_filter_params

        self.filters["uploadedarchive"].queryset = UploadedArchive.objects.filter(
            **user_has_access_to_uploadedarchives_filter_params(caiduser)
        ).order_by("-uploaded_at")


class NotificationFilter(django_filters.FilterSet):
    level = django_filters.NumberFilter(field_name="level")
    read = django_filters.BooleanFilter(field_name="read")
    created_after = django_filters.DateTimeFilter(field_name="created", lookup_expr="gte")
    created_before = django_filters.DateTimeFilter(field_name="created", lookup_expr="lte")

    search = django_filters.CharFilter(method="filter_search", label="Search")

    class Meta:
        model = models.Notification
        fields = ["level", "read"]

    def filter_search(self, queryset, name, value):
        # Annotate the queryset with a computed 'search' field.
        queryset = queryset.annotate(
            search=Concat(
                "name",
                Value(" "),
                "area__name",
            )
        )
        # Now filter on the annotated 'search' field.
        return queryset.filter(search__icontains=value)

        # fields = {
        #     "name": ['icontains'],
        #     "code": ['icontains'],
        #     "juv_code": ['icontains'],
        #     "sex": ["exact"],
        #     "coat_type": ["exact"],
        # }
