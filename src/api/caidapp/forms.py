from django import forms

from .models import MediaFile, UploadedArchive


class UploadedArchiveForm(forms.ModelForm):
    class Meta:
        model = UploadedArchive
        fields = ("archivefile", "location_at_upload")


class MediaFileForm(forms.ModelForm):
    class Meta:
        model = MediaFile
        fields = ("category", "location")
