from django import forms
from django.contrib.auth import get_user_model

from . import models
from .models import Album, CaIDUser, IndividualIdentity, MediaFile, UploadedArchive
import logging


logger = logging.getLogger(__name__)
User = get_user_model()

class UserIdentificationModelForm(forms.Form):
    identification_model = forms.ModelChoiceField(
        queryset=models.IdentificationModel.objects.all(), required=True
    )


class WorkgroupUsersForm(forms.Form):
    workgroup_users = forms.ModelMultipleChoiceField(
        queryset=CaIDUser.objects.all(), required=False
    )


# class MergeIdentityForm(forms.Form):
#     queryset = IndividualIdentity.objects.filter()
#     models.get_content_owner_filter_params()
#     identity = forms.ModelChoiceField(queryset=IndividualIdentity.objects.all(), required=False)
#
#     def __init__(self, *args, **kwargs):
#         super(MergeIdentityForm, self).__init__(*args, **kwargs)
#         self.fields["identity"].queryset = self.queryset


class TaxonForm(forms.ModelForm):
    class Meta:
        model = models.Taxon
        fields = ("name", "parent")


class AlbumForm(forms.ModelForm):
    class Meta:
        model = Album
        fields = ("name", "description")


class LocalityForm(forms.ModelForm):
    class Meta:
        model = models.Locality
        fields = ("name", "visible_name", "location")


class IndividualIdentityForm(forms.ModelForm):
    class Meta:
        model = IndividualIdentity
        fields = ("name", "code", "juv_code", "sex", "coat_type", "note", "birth_date", "death_date")

    birth_date = forms.DateField(
        widget=forms.TextInput(attrs={'placeholder': 'YYYY-MM-DD'}),
        required=False,
    )
    death_date = forms.DateField(
        widget=forms.TextInput(attrs={'placeholder': 'YYYY-MM-DD'}),
        required=False,
    )

class MergeIdentitiesForm(forms.Form):
    class Meta:
        model = IndividualIdentity
        fields = ("name", "code", "juv_code", "sex", "coat_type", "note", "birth_date", "death_date")

    birth_date = forms.DateField(
        widget=forms.TextInput(attrs={'placeholder': 'YYYY-MM-DD'}),
        required=False,
    )
    death_date = forms.DateField(
        widget=forms.TextInput(attrs={'placeholder': 'YYYY-MM-DD'}),
        required=False,
    )


class UploadedArchiveSelectTaxonForIdentificationForm(forms.ModelForm):
    class Meta:
        model = UploadedArchive
        fields = ("taxon_for_identification",)

class IndividualIdentitySelectSecondForMergeForm(forms.Form):
    def __init__(self, *args, identities=None, **kwargs):
        super().__init__(*args, **kwargs)
        if identities is not None:
            logger.debug(f"identities: {identities}")
            self.fields['identity'] = forms.ModelChoiceField(queryset=identities, required=True)


class UploadedArchiveUpdateBySpreadsheetForm(forms.Form):

    def __init__(self, *args, upload_to=None, **kwargs):
        super(UploadedArchiveUpdateBySpreadsheetForm, self).__init__(*args, **kwargs)

        # take only CSV or XLSX
        self.fields['spreadsheet_file'] = forms.FileField(label="Spreadsheet File",
                                                            required=True,
                                                            help_text="Select a CSV or XLSX file with the following columns: "
                                                                      "mediafile, locality_at_upload, locality_check_at, taxon_for_identification",
                                                            widget=forms.FileInput(attrs={"accept": ".csv,.xlsx"}),

            # "Spreadsheet File"
                                                          # upload_to=upload_to
                                                          )


class UploadedArchiveUpdateForm(forms.ModelForm):

    from .models import UploadedArchive

    locality_at_upload = forms.CharField(
        widget=forms.TextInput(attrs={"class": "autocomplete"}), required=False
    )

    class Meta:
        model = UploadedArchive
        fields = (
            # "archivefile",
            "name",
            "locality_at_upload",
            "locality_check_at"
            # "contains_identities"
        )


class UploadedArchiveForm(forms.ModelForm):

    locality_at_upload = forms.CharField(
        widget=forms.TextInput(attrs={"class": "autocomplete"}), required=False
    )
    locality_check_at = forms.DateField(
        widget=forms.DateInput(
            attrs={"class": "datepicker", "placeholder": "yyyy-mm-dd"}, format="%Y-%m-%d"
        ),
        input_formats=["%Y-%m-%d"],
        # widget=forms.TextInput(attrs={'class': 'datepicker'})
    )

    class Meta:
        model = UploadedArchive
        fields = ("archivefile", "locality_at_upload", "locality_check_at")
        help_texts = {
            "archivefile": "Select a zip file. Date and locality should be detected automatically, "
            "e.g., '2023-02-21_Horni Lukavice.zip', 'Horni Lukavice 20230221.zip'",
        }
        labels = {
            "archivefile": "Upload Archive File",
            "locality_at_upload": "Locality at Upload",
            "locality_check_at": "Locality Check Date",
        }


class UploadedArchiveFormWithTaxon(forms.ModelForm):

    locality_at_upload = forms.CharField(
        widget=forms.TextInput(attrs={"class": "autocomplete"}), required=False
    )

    class Meta:
        model = UploadedArchive
        fields = ("archivefile", "locality_at_upload", "taxon_for_identification")


class MediaFileForm(forms.ModelForm):
    class Meta:
        model = MediaFile
        fields = ("category", "locality", "identity", "captured_at", "taxon_verified", "note")


class MediaFileBulkForm(forms.ModelForm):
    # select_all = forms.BooleanField(required=False)
    class Meta:
        model = MediaFile
        fields = ("category", "identity", "identity_is_representative", "taxon_verified")

    def __init__(self, *args, **kwargs):
        super(MediaFileBulkForm, self).__init__(*args, **kwargs)
        self.fields["category"].queryset = models.Taxon.objects.order_by("name")


class MediaFileSelectionForm(forms.ModelForm):
    selected = forms.BooleanField(
        widget=forms.CheckboxInput(attrs={"class": "select-mediafile-checkbox"}), initial=False, required=False,
    )

    class Meta:
        model = MediaFile
        fields = ()


class MediaFileSetQueryForm(forms.Form):
    query = forms.CharField(max_length=100, required=False)
    pagenumber = forms.IntegerField(widget=forms.HiddenInput(), initial=1)
    filter_show_videos = forms.BooleanField(label="Show videos", initial=True, required=False)
    filter_show_images = forms.BooleanField(label="Show images", initial=True, required=False)
    filter_hide_empty = forms.BooleanField(label="Hide empty", initial=True, required=False)


class ChangeMediaFilesTimeForm(forms.Form):
    change_by_hours = forms.FloatField(label="Change by hours", required=False)
    change_by_days = forms.FloatField(label="Change by days", required=False)
    change_by_years = forms.FloatField(label="Change by years", required=False)


class LocalityImportForm(forms.Form):
    spreadsheet_file = forms.FileField()


class UploadedArchiveFilterForm:
    pass


class UserSelectForm(forms.Form):
    user = forms.ModelChoiceField(
        queryset=User.objects.all().order_by("username"), label="Select User"
    )
