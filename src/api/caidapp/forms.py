from django import forms

from . import models
from .models import Album, CIDUser, IndividualIdentity, MediaFile, UploadedArchive


class WorkgroupUsersForm(forms.Form):
    workgroup_users = forms.ModelMultipleChoiceField(queryset=CIDUser.objects.all(), required=False)


class AlbumForm(forms.ModelForm):
    class Meta:
        model = Album
        fields = ("name", "description")


class IndividualIdentityForm(forms.ModelForm):
    class Meta:
        model = IndividualIdentity
        fields = ("name",)


class UploadedArchiveForm(forms.ModelForm):
    class Meta:
        model = UploadedArchive
        fields = ("archivefile", "location_at_upload", "contains_identities")


class MediaFileForm(forms.ModelForm):
    class Meta:
        model = MediaFile
        fields = ("category", "location")


class MediaFileBulkForm(forms.ModelForm):
    class Meta:
        model = MediaFile
        fields = (
            "category",
            "identity",
            "identity_is_representative",
        )

    def __init__(self, *args, **kwargs):
        super(MediaFileBulkForm, self).__init__(*args, **kwargs)
        self.fields["category"].queryset = models.Taxon.objects.order_by("name")


class MediaFileSelectionForm(forms.ModelForm):
    selected = forms.BooleanField(initial=False, required=False)

    class Meta:
        model = MediaFile
        fields = ()


class MediaFileSetQueryForm(forms.Form):
    query = forms.CharField(max_length=100, required=False)
    pagenumber = forms.IntegerField(widget=forms.HiddenInput(), initial=1)
