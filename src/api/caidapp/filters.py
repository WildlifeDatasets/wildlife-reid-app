import logging
import re

import django_filters
from django.db.models import Q, Value
from django.db.models.functions import Concat

from . import models
from .models import Taxon, UploadedArchive

logger = logging.getLogger(__name__)


def _normalize_postgres_regex(pattern: str) -> str:
    """Translate common regex digit shorthands to PostgreSQL-compatible syntax."""
    return pattern.replace(r"\d", "[0-9]").replace(r"\D", "[^0-9]")


class LocalityFilter(django_filters.FilterSet):
    search = django_filters.CharFilter(method="filter_search", label="Search")
    area = django_filters.ModelChoiceFilter(queryset=models.Area.objects.all().order_by("name"))

    class Meta:
        model = models.Locality
        fields = {
            "name": ["icontains"],
            # "area": ['exact'],
        }

    # def filter_search(self, queryset, name, value):
    #     """Search across 'name' and related 'area__name' fields."""
    #     # Annotate the queryset with a computed 'search' field.
    #     queryset = queryset.annotate(
    #         search=Concat(
    #             "name",
    #             Value(" "),
    #             "area__name",
    #         )
    #     )
    #     # Now filter on the annotated 'search' field.
    #     return queryset.filter(search__icontains=value)

    def filter_search(self, queryset, name, value):
        """Search in taxon name, locality name, and identity name."""
        # Annotate the queryset with a computed 'search' field.
        if not value:
            return queryset
        return queryset.filter(
            Q(name__icontains=value)
            | Q(area__name__icontains=value)
            # | Q(observations__identity__name__icontains=value)
            # | Q(observations__taxon__name__icontains=value)
            # | Q(observations__identity__name__icontains=value)
        )


class IndividualIdentityFilter(django_filters.FilterSet):
    mediafile_count = django_filters.RangeFilter(label="Count of Media Files")

    last_seen = django_filters.DateFromToRangeFilter(
        label="Last Seen",
        help_text="Enter dates in YYYY-MM-DD format",
        widget=django_filters.widgets.RangeWidget(attrs={"type": "date"}),
    )
    search = django_filters.CharFilter(method="filter_search", label="Search")
    search_regex = django_filters.BooleanFilter(method="filter_search_regex", label="Use regex")
    orientation = django_filters.ChoiceFilter(
        field_name="observations__orientation",
        choices=models.ORIENTATION_CHOICES,
        label="Orientation",
    )
    identity_is_representative = django_filters.BooleanFilter(
        field_name="observations__identity_is_representative",
        label="Representative identity",
    )
    taxon_verified = django_filters.BooleanFilter(
        field_name="observations__taxon_verified",
        label="Taxon verified",
    )

    class Meta:
        model = models.IndividualIdentity
        fields = {
            # "name": ["icontains"],
            # "code": ["icontains"],
            # "juv_code": ["icontains"],
            "sex": ["exact"],
            "coat_type": ["exact"],
        }

    def filter_search(self, queryset, name, value):
        """Search in name, code, and juv_code fields."""
        # Annotate the queryset with a computed 'search' field.
        logger.debug(f"Filtering IndividualIdentity with search value: {value}")
        if not value:
            return queryset
        if self.data.get("search_regex"):
            return queryset
        return queryset.filter(
            Q(name__icontains=value)
            | Q(code__icontains=value)
            | Q(juv_code__icontains=value)
            # | Q(observations__identity__name__icontains=value)
            # | Q(observations__taxon__name__icontains=value)
            # | Q(observations__identity__name__icontains=value)
        )
        # queryset = queryset.annotate(
        #     search=Concat(
        #         "name",
        #         Value(" "),
        #         "code",
        #         Value(" "),
        #         "juv_code",
        #     )
        # )
        # # Now filter on the annotated 'search' field.
        # return queryset.filter(search__icontains=value)

    def filter_search_regex(self, queryset, name, value):
        """Apply regex search to name, code, and juv_code when explicitly enabled."""
        search_value = self.data.get("search")
        if not value or not search_value:
            return queryset
        try:
            re.compile(search_value)
        except re.error:
            return queryset.none()
        db_regex = _normalize_postgres_regex(search_value)
        return queryset.filter(
            Q(name__iregex=db_regex)
            | Q(code__iregex=db_regex)
            | Q(juv_code__iregex=db_regex)
        )


class MediaFileFilter(django_filters.FilterSet):
    # A free-text search filter. This uses a custom method to apply full text search.

    request = None
    # taxon = django_filters.ModelChoiceFilter(queryset=models.Taxon.objects.all().order_by("name"))
    # taxon from observations__taxon, but only those that are in the same workgroup as the user

    taxon = django_filters.ModelChoiceFilter(
        queryset=Taxon.objects.all().order_by("name"),
        label="Taxon",
        field_name="observations__taxon",
    )
    uploadedarchive = django_filters.ModelChoiceFilter(
        field_name="parent",
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
    search_regex = django_filters.BooleanFilter(method="filter_search_regex", label="Use regex")

    class Meta:
        model = models.MediaFile
        # Declare the fields you want to filter by.
        taxon = Taxon.objects.all().order_by("name")
        fields = {
            "media_type": ["exact"],
        }

    def filter_search(self, queryset, name, value):
        """Search in taxon name, locality name, and identity name."""
        # Annotate the queryset with a computed 'search' field.
        if not value:
            return queryset
        if self.data.get("search_regex"):
            return queryset
        return queryset.filter(
            Q(locality__name__icontains=value)
            | Q(observations__taxon__name__icontains=value)
            | Q(observations__identity__name__icontains=value)
            | Q(original_filename__icontains=value)
            | Q(mediafile__icontains=value)
        )
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

    def filter_search_regex(self, queryset, name, value):
        """Apply regex search to the same fields as plain text search when explicitly enabled."""
        search_value = self.data.get("search")
        if not value or not search_value:
            return queryset
        try:
            re.compile(search_value)
        except re.error:
            return queryset.none()
        db_regex = _normalize_postgres_regex(search_value)
        return queryset.filter(
            Q(locality__name__iregex=db_regex)
            | Q(observations__taxon__name__iregex=db_regex)
            | Q(observations__identity__name__iregex=db_regex)
            | Q(original_filename__iregex=db_regex)
            | Q(mediafile__iregex=db_regex)
        )

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
        """Search in title and message fields."""
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
