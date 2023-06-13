from django import forms

from .models import UploadedArchive


class UploadedArchiveForm(forms.ModelForm):
    class Meta:
        model = UploadedArchive
        fields = ("archivefile", "location_at_upload")
