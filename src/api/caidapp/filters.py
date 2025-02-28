import django_filters


from . import models


class LocalityFilter(django_filters.FilterSet):
    class Meta:
        model = models.Locality
        fields = {
            "name": ['icontains'],
            "area": ['exact'],
        }


class IndividualIdentityFilter(django_filters.FilterSet):
    mediafile_count = django_filters.RangeFilter(label="Count of Media Files")

    last_seen = django_filters.DateFromToRangeFilter(
        label="Last Seen",
        help_text="Enter dates in YYYY-MM-DD format",
        widget=django_filters.widgets.RangeWidget(attrs={'type': 'date'})
    )
    class Meta:
        model = models.IndividualIdentity
        fields = {
            "name": ['icontains'],
            "code": ['icontains'],
            "juv_code": ['icontains'],
            "sex": ["exact"],
            "coat_type": ["exact"],
        }