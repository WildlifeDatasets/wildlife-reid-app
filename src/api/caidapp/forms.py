import logging

from django import forms
from django.contrib.auth import get_user_model

from . import models
from .models import (
    Album,
    AnimalObservation,
    CaIDUser,
    IndividualIdentity,
    Locality,
    MediaFile,
    UploadedArchive,
    WorkGroup,
)

logger = logging.getLogger(__name__)
User = get_user_model()


class SmallTextarea(forms.Textarea):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("attrs", {})
        kwargs["attrs"].setdefault("rows", 3)
        super().__init__(*args, **kwargs)


# Nastav globálně jako výchozí Textarea
forms.Textarea = SmallTextarea


class CompareLocalitiesForm(forms.Form):
    locality = forms.ModelChoiceField(queryset=Locality.objects.all(), label="Locality")


class UserIdentificationModelForm(forms.Form):
    identification_model = forms.ModelChoiceField(queryset=models.IdentificationModel.objects.all(), required=True)


# deprecated TODO remove
# class WorkgroupUsersForm(forms.Form):
#     workgroup_users = forms.ModelMultipleChoiceField(queryset=CaIDUser.objects.all(), required=False)


# class WorkgroupForm(forms.ModelForm):
#     class Meta:
#         model = WorkGroup
#         fields = ["name", 'default_taxon_for_identification', 'caiduser_set']   # nebo jiná pole, která chceš editovat


class WorkgroupForm(forms.ModelForm):
    caidusers = forms.ModelMultipleChoiceField(
        queryset=CaIDUser.objects.all(),
        # widget=forms.CheckboxSelectMultiple,  # nebo forms.SelectMultiple
        widget=forms.SelectMultiple,
        required=False,
    )

    class Meta:
        model = WorkGroup
        fields = [
            "name",
            "default_taxon_for_identification",
            "sequence_time_limit",
            "check_taxon_before_identification",
            "caidusers",
        ]
        help_texts = {
            "check_taxon_before_identification": "Do the identification only for media files "
            + "and observations with the correct taxon. "
            + "Ignore the other observations and media files.",
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance.pk:
            self.fields["caidusers"].initial = self.instance.caiduser_set.all().order_by("user__username")

    def save(self, commit=True):
        """Save the WorkGroup and update the related CaIDUser instances."""
        workgroup = super().save(commit=commit)
        if commit:
            workgroup.caiduser_set.set(self.cleaned_data["caidusers"])
        return workgroup


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


# class CaIDForm(forms.ModelForm):
#     class Meta:
#         model = models.CaIDUser
#         fields = ("default_taxon_for_identification", "timezone", "ml_consent_given", )

# widgets = {
#     'locality': forms.TextInput(attrs={'class': 'autocomplete'}),
# }


class WellcomeForm(forms.ModelForm):
    class Meta:
        model = CaIDUser
        fields = ("show_taxon_classification", "show_wellcome_message_on_next_login")


class CaIDUserSettingsForm(forms.ModelForm):
    class Meta:
        model = CaIDUser
        fields = (
            "show_taxon_classification",
            "default_taxon_for_identification",
            "timezone",
            "ml_consent_given",
            "show_wellcome_message_on_next_login",
        )


class AlbumForm(forms.ModelForm):
    class Meta:
        model = Album
        fields = ("name", "description")


class LocalityForm(forms.ModelForm):
    class Meta:
        model = models.Locality
        fields = ("name", "visible_name", "location", "note")


class IndividualIdentityForm(forms.ModelForm):
    class Meta:
        model = IndividualIdentity
        fields = (
            "name",
            "code",
            "juv_code",
            "sex",
            "coat_type",
            "note",
            "birth_date",
            "death_date",
        )

        widgets = {
            "birth_date": forms.DateInput(attrs={"type": "date"}),
            "death_date": forms.DateInput(attrs={"type": "date"}),
        }

    # birth_date = forms.DateField(
    #     widget=forms.TextInput(attrs={'type': 'date'}),
    #     required=False,
    # )
    # death_date = forms.DateField(
    #     widget=forms.TextInput(attrs={'placeholder': 'YYYY-MM-DD'}),
    #     required=False,
    # )


class MergeIdentitiesForm(forms.Form):
    class Meta:
        model = IndividualIdentity
        fields = (
            "name",
            "code",
            "juv_code",
            "sex",
            "coat_type",
            "note",
            "birth_date",
            "death_date",
        )

        widgets = {
            "birth_date": forms.DateInput(attrs={"type": "date"}),
            "death_date": forms.DateInput(attrs={"type": "date"}),
        }

    # birth_date = forms.DateField(
    #     widget=forms.TextInput(attrs={'placeholder': 'YYYY-MM-DD'}),
    #     required=False,
    # )
    # death_date = forms.DateField(
    #     widget=forms.TextInput(attrs={'placeholder': 'YYYY-MM-DD'}),
    #     required=False,
    # )


class UploadedArchiveSelectTaxonForIdentificationForm(forms.ModelForm):
    taxon_for_identification = forms.ModelChoiceField(
        queryset=models.Taxon.objects.all().order_by("name"), required=True
    )

    class Meta:
        model = UploadedArchive
        fields = ("taxon_for_identification",)


class IndividualIdentitySelectSecondForMergeForm(forms.Form):
    def __init__(self, *args, identities=None, **kwargs):
        super().__init__(*args, **kwargs)
        if identities is not None:
            logger.debug(f"identities: {identities}")
            self.fields["identity"] = forms.ModelChoiceField(queryset=identities, required=True)


class UploadedArchiveUpdateBySpreadsheetForm(forms.Form):

    def __init__(self, *args, upload_to=None, **kwargs):
        super(UploadedArchiveUpdateBySpreadsheetForm, self).__init__(*args, **kwargs)

        # take only CSV or XLSX
        self.fields["spreadsheet_file"] = forms.FileField(
            label="Spreadsheet File",
            required=True,
            help_text="Select a CSV or XLSX file with the following columns: "
            "mediafile, locality_at_upload, locality_check_at, taxon_for_identification",
            widget=forms.FileInput(attrs={"accept": ".csv,.xlsx"}),
            # "Spreadsheet File"
            # upload_to=upload_to
        )


class UploadedArchiveUpdateForm(forms.ModelForm):

    from .models import UploadedArchive

    locality_at_upload = forms.CharField(widget=forms.TextInput(attrs={"class": "autocomplete"}), required=False)

    class Meta:
        model = UploadedArchive
        fields = (
            # "archivefile",
            "name",
            "locality_at_upload",
            "locality_check_at",
            # "contains_identities"
        )
class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True


class MultipleFileField(forms.FileField):

    def __init__(self, *args, **kwargs):
        # Nastavíme náš vlastní widget automaticky
        kwargs.setdefault("widget", MultipleFileInput())
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        # nejdřív použijeme defaultní validaci FileFieldu
        single_clean = super().clean

        # pokud je data list/tuple → zvalidujeme každý soubor
        if isinstance(data, (list, tuple)):
            return [single_clean(d, initial) for d in data]

        # jinak zpracujeme jako jeden soubor
        return single_clean(data, initial)

class UploadedArchiveForm(forms.ModelForm):

    # archivefile = forms.FileField(
    #     widget=forms.FileInput(
    #         # attrs={'multiple': True}
    #     ),
    #     required=True,
    #     label="Upload files",
    #     help_text=(
    #         "Select one or more files. If multiple files are uploaded, "
    #         "they will automatically be zipped before processing."
    #     ),
    # )
    archivefile = MultipleFileField(
        required=True,
        label="Upload files",
        help_text="Select files; multiple files will be zipped."
    )

    locality_at_upload = forms.CharField(
        label="Locality",
        widget=forms.TextInput(attrs={"class": "autocomplete"}),
        required=False,
    )

    ml_consent = forms.BooleanField(
        widget=forms.CheckboxInput(),
        label="I agree to the use of my uploaded images and videos for training AI models.",
        required=True,
    )

    locality_check_at = forms.DateField(
        widget=forms.DateInput(
            attrs={"class": "datepicker", "placeholder": "yyyy-mm-dd"},
            format="%Y-%m-%d",
        ),
        input_formats=["%Y-%m-%d"],
    )

    class Meta:
        model = UploadedArchive
        fields = ("locality_at_upload", "locality_check_at")
        exclude = ("archivefile",)     # ← 🔥 přidat sem
        labels = {
            "locality_at_upload": "Locality at Upload",
            "locality_check_at": "Locality Check Date",
        }

    def __init__(self, *args, **kwargs):
        user = kwargs.pop("user", None)
        super().__init__(*args, **kwargs)
        self.fields["ml_consent"].initial = user.caiduser.ml_consent_given if user else False


# class UploadedArchiveForm(forms.ModelForm):
#
#     # 🔥 vlastní pole mimo Meta — to je klíčové
#     archivefile = forms.FileField(
#         widget=forms.FileInput(attrs={'multiple': True}),
#         required=True,
#         label="Upload files",
#         help_text=(
#             "Select one or more files. If multiple files are uploaded, "
#             "they will automatically be zipped before processing."
#         ),
#     )
#
#     locality_at_upload = forms.CharField(
#         widget=forms.TextInput(attrs={"class": "autocomplete"}),
#         required=False,
#     )
#
#     ml_consent = forms.BooleanField(
#         widget=forms.CheckboxInput(),
#         label="I agree to the use of my uploaded images and videos for training AI models.",
#         required=True,
#     )
#
#     locality_check_at = forms.DateField(
#         widget=forms.DateInput(
#             attrs={"class": "datepicker", "placeholder": "yyyy-mm-dd"},
#             format="%Y-%m-%d",
#         ),
#         input_formats=["%Y-%m-%d"],
#     )
#
#     class Meta:
#         model = UploadedArchive
#
#         # 🔥 ARCHIVEFILE NESMÍ BÝT V fields
#         fields = ("locality_at_upload", "locality_check_at")
#
#         # 🔥 taktéž nesmíš mít help_texts nebo labels pro archivefile
#         labels = {
#             "locality_at_upload": "Locality at Upload",
#             "locality_check_at": "Locality Check Date",
#         }
#
#     def __init__(self, *args, **kwargs):
#         user = kwargs.pop("user", None)
#         super().__init__(*args, **kwargs)
#         self.fields["ml_consent"].initial = user.caiduser.ml_consent_given if user else False


class UploadedArchiveFormWithTaxon(forms.ModelForm):

    # archivefile = forms.FileField(
    #     widget=forms.FileInput(
    #         # attrs={'multiple': True}
    #     ),
    #     required=True,
    #     label="Upload files",
    # )
    archivefile = MultipleFileField(
        required=True,
        label="Upload files",
        help_text="Select files; multiple files will be zipped."
    )

    locality_at_upload = forms.CharField(
        label="Locality",
        widget=forms.TextInput(attrs={"class": "autocomplete"}),
        required=False
    )

    taxon_for_identification = forms.ModelChoiceField(
        queryset=models.Taxon.objects.all().order_by("name"),
        required=True,
    )

    ml_consent = forms.BooleanField(
        widget=forms.CheckboxInput(),
        label="I agree to the use of my uploaded images and videos for training AI models.",
        required=True,
    )

    class Meta:
        model = UploadedArchive

        fields = ("locality_at_upload", "taxon_for_identification")
        exclude = ("archivefile",)   # <– pořád nutné!

        # again NO archivefile here


    def __init__(self, *args, **kwargs):
        user = kwargs.pop("user", None)
        super().__init__(*args, **kwargs)
        self.fields["ml_consent"].initial = user.caiduser.ml_consent_given if user else False

# class UploadedArchiveForm(forms.ModelForm):
#
#     locality_at_upload = forms.CharField(widget=forms.TextInput(attrs={"class": "autocomplete"}), required=False)
#     ml_consent = forms.BooleanField(
#         widget=forms.CheckboxInput(),
#         label="I agree to the use of my uploaded images and videos for training AI models.",
#         required=True,
#     )
#     locality_check_at = forms.DateField(
#         widget=forms.DateInput(
#             attrs={"class": "datepicker", "placeholder": "yyyy-mm-dd"},
#             format="%Y-%m-%d",
#         ),
#         input_formats=["%Y-%m-%d"],
#         # widget=forms.TextInput(attrs={'class': 'datepicker'})
#     )
#     archivefile = forms.FileField(
#         widget=forms.FileInput(attrs={'multiple': True}),
#         required=True
#     )
#
#     class Meta:
#         model = UploadedArchive
#         fields = (
#             # "archivefile",
#                   "locality_at_upload", "locality_check_at")
#         help_texts = {
#             "archivefile": "Select a zip file. Date and locality should be detected automatically, "
#             "e.g., '2023-02-21_Horni Lukavice.zip', 'Horni Lukavice 20230221.zip'",
#         }
#         labels = {
#             "archivefile": "Upload Archive File",
#             "locality_at_upload": "Locality at Upload",
#             "locality_check_at": "Locality Check Date",
#         }
#         # widgets = {
#         #     "archivefile": forms.FileInput(attrs={"multiple": True}),
#         # }
#
#     def __init__(self, *args, **kwargs):
#         user = kwargs.pop("user", None)
#         super().__init__(*args, **kwargs)
#         self.fields["ml_consent"].initial = user.caiduser.ml_consent_given if user else False
#         # if user and user.caiduser.ml_consent_given:
#         #     # Don't show the checkbox if already agreed
#         #     self.fields.pop("ml_consent")


# class UploadedArchiveFormWithTaxon(forms.ModelForm):
#
#     locality_at_upload = forms.CharField(widget=forms.TextInput(attrs={"class": "autocomplete"}), required=False)
#     taxon_for_identification = forms.ModelChoiceField(
#         queryset=models.Taxon.objects.all().order_by("name"), required=True
#     )
#
#     ml_consent = forms.BooleanField(
#         label="I agree to the use of my uploaded images and videos for training AI models.",
#         required=True,
#     )
#     archivefile = forms.FileField(
#         widget=forms.FileInput(attrs={'multiple': True}),
#         required=True
#     )
#
#     class Meta:
#         model = UploadedArchive
#         # widgets = {
#         #     "archivefile": forms.FileInput(attrs={"multiple": True}),
#         # }
#         fields = (
#             # "archivefile",
#             "locality_at_upload",
#         )
#
#     def __init__(self, *args, **kwargs):
#         user = kwargs.pop("user", None)
#         super().__init__(*args, **kwargs)
#         self.fields["ml_consent"].initial = user.caiduser.ml_consent_given if user else False
#         # if user and user.caiduser.ml_consent_given:
#         #     # Don't show the checkbox if already agreed
#         #     self.fields.pop("ml_consent")



class CaIDUserForm(forms.ModelForm):
    class Meta:
        model = CaIDUser
        fields = ("show_taxon_classification",)

        help_texts = {
            "show_taxon_classification": "Do you plan to use taxon classification?",
        }


class MediaFileForm(forms.ModelForm):
    class Meta:
        model = MediaFile
        fields = (
            # "taxon",
            # "taxon_verified",
            "locality",
            # "identity",
            # "identity_is_representative",
            "captured_at",
            "note",
            # "orientation",
        )

    def __init__(self, *args, **kwargs):
        mediafile = kwargs.get("instance")
        super().__init__(*args, **kwargs)
        # Only show the identities accessible to the given user.
        caiduser = mediafile.parent.owner
        if "identity" in self.fields:
            if caiduser.workgroup is not None:
                self.fields["identity"].queryset = IndividualIdentity.objects.filter(
                    # adjust this filter to however your user-Identity relationship is defined
                    owner_workgroup=caiduser.workgroup
                )
            else:
                # fields identity is empty
                self.fields["identity"].queryset = IndividualIdentity.objects.none()

        self.fields["locality"].queryset = models.Locality.objects.filter(
            **models.user_has_access_filter_params(caiduser, "owner")
        ).order_by("name")
        # self.fields["taxon"].queryset = models.Taxon.objects.order_by("name")


class MediaFileMissingTaxonForm(forms.ModelForm):
    class Meta:
        model = MediaFile
        fields = ("taxon", "taxon_verified", "locality")

    def __init__(self, *args, **kwargs):
        mediafile = kwargs.get("instance")
        super().__init__(*args, **kwargs)
        # Only show the identities accessible to the given user.
        caiduser = mediafile.parent.owner

        self.fields["locality"].queryset = models.Locality.objects.filter(
            **models.user_has_access_filter_params(caiduser, "owner")
        ).order_by("name")
        self.fields["taxon"].queryset = models.Taxon.objects.order_by("name")


class MediaFileBulkForm(forms.ModelForm):
    # select_all = forms.BooleanField(required=False)
    class Meta:
        model = MediaFile
        fields = ("taxon", "identity", "identity_is_representative", "taxon_verified")

    def __init__(self, *args, **kwargs):
        super(MediaFileBulkForm, self).__init__(*args, **kwargs)
        self.fields["taxon"].queryset = models.Taxon.objects.order_by("name")


class MediaFileSelectionForm(forms.ModelForm):
    selected = forms.BooleanField(
        widget=forms.CheckboxInput(attrs={"class": "select-mediafile-checkbox"}),
        initial=False,
        required=False,
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
    filter_orientation = forms.ChoiceField(
        label="Orientation",
        choices=(
            ("All", "All"),
            ("L", "Left"),
            ("R", "Right"),
            ("F", "Front"),
            ("B", "Back"),
            ("U", "Unknown"),
        ),
        initial="all",
        required=False,
    )


class ChangeMediaFilesTimeForm(forms.Form):
    change_by_hours = forms.FloatField(label="Change by hours", required=False)
    change_by_days = forms.FloatField(label="Change by days", required=False)
    change_by_years = forms.FloatField(label="Change by years", required=False)


class SpreadsheetFileImportForm(forms.Form):
    spreadsheet_file = forms.FileField()


class UploadedArchiveFilterForm:
    pass


class UserSelectForm(forms.Form):
    user = forms.ModelChoiceField(queryset=User.objects.all().order_by("username"), label="Select User")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fields["user"].label_from_instance = lambda obj: (
            f"{obj.first_name} {obj.last_name}".strip() if obj.first_name or obj.last_name else obj.username
        )


class ColumnMappingForm(forms.Form):
    original_path = forms.ChoiceField(choices=[], required=True)
    unique_name = forms.ChoiceField(choices=[], required=False)
    taxon = forms.ChoiceField(choices=[], required=False)
    locality_name = forms.ChoiceField(choices=[], required=False)
    datetime = forms.ChoiceField(choices=[], required=False)
    latitude = forms.ChoiceField(choices=[], required=False)
    longitude = forms.ChoiceField(choices=[], required=False)

    def __init__(self, *args, **kwargs):
        column_choices = kwargs.pop("column_choices", [])
        super().__init__(*args, **kwargs)

        choices = [("", "----")] + [(col, col) for col in column_choices]  # možnost nevybrat

        for field_name in self.fields:
            self.fields[field_name].choices = choices

        # Předvyplnění pokud název sloupce odpovídá očekávanému jménu
        for field_name in self.fields:
            if field_name in column_choices:
                self.initial[field_name] = field_name


class AnimalObservationForm(forms.ModelForm):
    class Meta:
        model = AnimalObservation
        fields = "__all__"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # caiduser = self.instance.mediafile.parent.owner
        if self.instance and self.instance.mediafile_id:
            workgroup = self.instance.mediafile.parent.owner.workgroup
            self.fields["identity"].queryset = self.fields["identity"].queryset.filter(owner_workgroup=workgroup)

        self.fields["taxon"].queryset = models.Taxon.objects.order_by("name")


# class AnimalObservationForm(forms.ModelForm):
#     class Meta:
#         model = AnimalObservation
#         fields = [
#             "taxon",
#             "identity", "identity_is_representative",
#             "bbox_x_center", "bbox_y_center", "bbox_width", "bbox_height",
#         ]
#
#         widgets = {
#             "bbox_x_center": HiddenInput(),
#             "bbox_y_center": HiddenInput(),
#             "bbox_width": HiddenInput(),
#             "bbox_height": HiddenInput(),
#         }

# def __init__(self, *args, **kwargs):
#     request = kwargs.pop("request", None)  # ← získáme request správně
#     super().__init__(*args, **kwargs)
#
#     # request = getattr(self, "request", None)
#     print("AnimalObservationForm __init__")
#     print(f"request: {request}")
#     if True:
#         # if request:
#         # next_url is the
#         # next_url = request.path
#         # we dont know the actual path, because we do not have request here
#         next_url = reverse("caidapp:media_file_update", args=[self.instance.mediafile.id])
#         create_url = reverse("caidapp:individual_identity_create",
#                              args=[self.instance.mediafile.id]) + f"?next={next_url}"
#         self.helper = FormHelper()
#         self.helper.layout = Layout(
#             "taxon",
#             Row(
#                 Column("identity", css_class="col-auto"),
#                 Column(
#                     HTML(f"""
#                         <a href="{create_url}" class="btn btn-outline-primary btn-sm" title="Add new identity">
#                             <i class="bi bi-plus"></i>
#                         </a>
#                     """),
#                     css_class="col-auto"
#                 ),
#             ),
#             "identity_is_representative",
#         )
