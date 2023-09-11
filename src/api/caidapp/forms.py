from django import forms

from .models import Album, MediaFile, UploadedArchive, CIDUser, IndividualIdentity

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
        fields = ("archivefile", "location_at_upload")


class MediaFileForm(forms.ModelForm):
    class Meta:
        model = MediaFile
        fields = ("category", "location")


class MediaFileBulkForm(forms.ModelForm):
    class Meta:
        model = MediaFile
        fields = ("category",)


class MediaFileSelectionForm(forms.ModelForm):
    selected = forms.BooleanField(initial=False, required=False)

    class Meta:
        model = MediaFile
        fields = ()


class MediaFileSetQueryForm(forms.Form):
    query = forms.CharField(max_length=100, required=False)
    pagenumber = forms.IntegerField(widget=forms.HiddenInput(), initial=1)
