from django import forms
from django.contrib.auth import get_user_model


from . import models
from .models import Album, CaIDUser, IndividualIdentity, MediaFile, UploadedArchive

User = get_user_model()

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
        fields = (
            "name",
            "parent"
        )

class AlbumForm(forms.ModelForm):
    class Meta:
        model = Album
        fields = ("name", "description")


class LocationForm(forms.ModelForm):
    class Meta:
        model = models.Location
        fields = ("name", "visible_name", "location")


class IndividualIdentityForm(forms.ModelForm):
    class Meta:
        model = IndividualIdentity
        fields = ("name", "code","juv_code", "sex", 'coat_type', 'note')


class UploadedArchiveSelectTaxonForIdentificationForm(forms.ModelForm):
    class Meta:
        model = UploadedArchive
        fields = ("taxon_for_identification",)

class UploadedArchiveUpdateForm(forms.ModelForm):
    from django import forms

    from .models import UploadedArchive

    location_at_upload = forms.CharField(
        widget=forms.TextInput(attrs={"class": "autocomplete"}), required=False
    )

    class Meta:
        model = UploadedArchive
        fields = (
            # "archivefile",
            'name',
            "location_at_upload",
            'location_check_at'
            # "contains_identities"
        )


class UploadedArchiveForm(forms.ModelForm):
    from django import forms

    from .models import UploadedArchive

    # archivefile = forms.FileField(attrs={"accept": ".zip", "placeholder": "Select a zip file (optinal format {YYYY-MM-DD}_{location}.zip)"})

    location_at_upload = forms.CharField(
        widget=forms.TextInput(attrs={"class": "autocomplete"}), required=False
    )
    location_check_at = forms.DateField(
        widget=forms.DateInput(attrs={'class': 'datepicker', 'placeholder': 'yyyy-mm-dd'}, format='%Y-%m-%d'),
        input_formats=['%Y-%m-%d'],

        # widget=forms.TextInput(attrs={'class': 'datepicker'})
    )

    class Meta:
        model = UploadedArchive
        fields = (
            "archivefile",
            "location_at_upload",
            'location_check_at'
        )
        help_texts = {
            'archivefile': "Select a zip file (optional format: YYYY-MM-DD_location.zip)",
        }
        labels = {
            'archivefile': 'Upload Archive File',
            'location_at_upload': 'Location at Upload',
            'location_check_at': 'Location Check Date',
        }


class UploadedArchiveFormWithTaxon(forms.ModelForm):

    location_at_upload = forms.CharField(
        widget=forms.TextInput(attrs={"class": "autocomplete"}), required=False
    )
    class Meta:
        model = UploadedArchive
        fields = (
            "archivefile",
            "location_at_upload",
            "taxon_for_identification"
        )



class MediaFileForm(forms.ModelForm):
    # captured_at = forms.DateField(
    #     widget=forms.DateInput(attrs={'class': 'datepicker', 'placeholder': 'yyyy-mm-dd'}, format='%Y-%m-%d'),
    #     input_formats=['%Y-%m-%d'],
    #
    #     # widget=forms.TextInput(attrs={'class': 'datepicker'})
    # )
    class Meta:
        model = MediaFile
        fields = ("category", "location", "identity", "captured_at", "taxon_verified")


class MediaFileBulkForm(forms.ModelForm):

    class Meta:
        model = MediaFile
        fields = (
            "category",
            "identity",
            "identity_is_representative",
            "taxon_verified"
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


class LocationImportForm(forms.Form):
    spreadsheet_file = forms.FileField()


class UploadedArchiveFilterForm():
    pass


class UserSelectForm(forms.Form):
    user = forms.ModelChoiceField(queryset=User.objects.all().order_by("username"), label="Select User")
