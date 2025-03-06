import django_filters
from django.db.models import Value, CharField
from django.db.models.functions import Concat, Cast

from . import models


class LocalityFilter(django_filters.FilterSet):
    search = django_filters.CharFilter(method='filter_search', label="Search")
    area = django_filters.ModelChoiceFilter(
        queryset=models.Area.objects.all().order_by('name')
    )
    class Meta:
        model = models.Locality
        fields = {
            "name": ['icontains'],
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
        widget=django_filters.widgets.RangeWidget(attrs={'type': 'date'})
    )
    search = django_filters.CharFilter(method='filter_search', label="Search")
    class Meta:
        model = models.IndividualIdentity
        fields = {
            "name": ['icontains'],
            "code": ['icontains'],
            "juv_code": ['icontains'],
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
from django.shortcuts import get_object_or_404
from django.contrib.postgres.search import SearchQuery, SearchRank, SearchVector
from django.db.models import Q
from .models import MediaFile, Album, IndividualIdentity, Taxon, UploadedArchive, Locality


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
    taxon = django_filters.ModelChoiceFilter(
        queryset=models.Taxon.objects.all().order_by('name')
    )
    uploadedarchive = django_filters.ModelChoiceFilter(
        queryset=models.UploadedArchive.objects.all()
            # .annotate(
            #     name_extended=Concat(
            #         'name',
            #         Value(' ('),
            #         Cast('uploaded_at', output_field=CharField()),
            #         Value(')')
            #     )
            # )
            .order_by('-uploaded_at'),
        # annotate(
            # name_extended=Concat('name', Value(' ( asd'),  Value(')'))
        # ).order_by('-uploaded_at'),
        # label_from_instance=lambda obj: f"{obj.name} ({obj.uploaded_at.strftime('%Y-%m-%d')})",
        label='Uploaded Archive',
        # label_from_instance=lambda obj: obj.name_extended
    )
    #
    captured_at = django_filters.DateFromToRangeFilter(
        label="Captured At",
        help_text="Enter dates in YYYY-MM-DD format",
        widget=django_filters.widgets.RangeWidget(attrs={'type': 'date'}),
    )
    search = django_filters.CharFilter(method='filter_search', label="Search")

    class Meta:
        model = models.MediaFile
        # Declare the fields you want to filter by.
        taxon = Taxon.objects.all().order_by('name')
        fields = {
            "locality__name": ['icontains'],
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

