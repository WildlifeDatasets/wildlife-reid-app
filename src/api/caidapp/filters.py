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
    class Meta:
        model = models.IndividualIdentity
        fields = {
            "name": ['icontains'],
            "code": ['icontains'],
            "juv_code": ['icontains'],
        }