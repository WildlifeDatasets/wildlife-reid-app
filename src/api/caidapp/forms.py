import logging
from pathlib import Path

from django import forms
from django.contrib.auth import get_user_model
from django.db.models import Q

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


class WorkGroupInvitationForm(forms.ModelForm):
    user_identifier = forms.CharField(
        label="Username or email",
        help_text="Enter the exact username or email address of the user.",
    )

    class Meta:
        model = models.WorkGroupInvitation
        fields = []

    def __init__(self, *args, target_workgroup=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_workgroup = target_workgroup

    def clean_user_identifier(self):
        identifier = self.cleaned_data["user_identifier"].strip()
        matches = models.CaIDUser.objects.filter(
            Q(user__username__iexact=identifier) | Q(user__email__iexact=identifier)
        ).distinct()
        if matches.count() != 1:
            raise forms.ValidationError("No unique user was found for that username or email.")

        invited_user = matches.get()
        if invited_user.workgroup_id == getattr(self.target_workgroup, "pk", None):
            raise forms.ValidationError("This user is already a member of the workgroup.")
        if models.WorkGroupInvitation.objects.filter(
            invited_user=invited_user,
            target_workgroup=self.target_workgroup,
        ).exists():
            raise forms.ValidationError("An invitation for this user and workgroup already exists.")
        self.instance.invited_user = invited_user
        return identifier


def normalize_regex_input(value: str) -> str:
    """Accept regex pasted as a Python string literal, e.g. r"...", from chat tools."""
    if not value:
        return value
    value = value.strip()
    prefix = value[:1].lower()
    quote_start = 1 if prefix == "r" and len(value) >= 2 and value[1] in ("'", '"') else 0
    if value[quote_start : quote_start + 1] not in ("'", '"'):
        return value
    quote = value[quote_start]
    if value.endswith(quote):
        return value[quote_start + 1 : -1]
    return value


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
            "identity_code_regex",
            "caidusers",
            "identification_model",
            "detection_model_path",
            "detection_model_architecture",
        ]
        help_texts = {
            "check_taxon_before_identification": "Do the identification only for media files "
            + "and observations with the correct taxon. "
            + "Ignore the other observations and media files.",
            "identity_code_regex": "Regex used to extract identity codes from identity names, for example B75.",
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
            "show_base_between_regular_uploads",
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
        # Automatically use our custom widget
        kwargs.setdefault("widget", MultipleFileInput())
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        """User-friendly validation for multiple files."""
        single_clean = super().clean

        # validate every file if multiple files are uploaded
        if isinstance(data, (list, tuple)):
            return [single_clean(d, initial) for d in data]

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
        required=True, label="Upload files", help_text="Select files; multiple files will be zipped."
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
        exclude = ("archivefile",)  # ← 🔥 přidat sem
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
        required=True, label="Upload files", help_text="Select files; multiple files will be zipped."
    )

    locality_at_upload = forms.CharField(
        label="Locality", widget=forms.TextInput(attrs={"class": "autocomplete"}), required=False
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
        exclude = ("archivefile",)  # <– pořád nutné!

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


class NewUploadForm(forms.Form):
    UPLOAD_TARGET_CHOICES = (
        ("taxon_processing", "Classify taxa first"),
        ("identification", "Use for identification / re-identification"),
    )
    TAXON_MODE_CHOICES = (
        ("recognize_taxa", "Recognize taxa"),
        ("single_taxon", "Single known taxon"),
    )

    upload_files = MultipleFileField(
        required=True,
        label="Upload files",
        help_text="Select a ZIP, media files, and optionally one spreadsheet.",
    )
    spreadsheet_file = forms.FileField(
        required=False,
        label="Spreadsheet",
        widget=forms.FileInput(attrs={"accept": ".csv,.xlsx"}),
    )
    locality_at_upload = forms.CharField(
        label="Locality",
        widget=forms.TextInput(attrs={"class": "autocomplete"}),
        required=False,
    )
    locality_check_at = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={"type": "date"}),
        input_formats=["%Y-%m-%d"],
    )
    upload_target = forms.ChoiceField(
        choices=UPLOAD_TARGET_CHOICES,
        required=False,
        widget=forms.RadioSelect,
        label="Where should this upload go?",
    )
    taxon_mode = forms.ChoiceField(choices=TAXON_MODE_CHOICES, initial="recognize_taxa", required=False)
    taxon_for_identification = forms.ModelChoiceField(
        queryset=models.Taxon.objects.all().order_by("name"),
        required=False,
    )
    contains_identities = forms.BooleanField(required=False)
    is_for_identification = forms.BooleanField(
        required=False,
        label="Use this upload for identification / re-identification",
    )
    directory_structure = forms.CharField(required=False, widget=forms.HiddenInput())
    directory_mapping = forms.CharField(required=False, widget=forms.HiddenInput())
    path_regex = forms.CharField(required=False, widget=forms.HiddenInput())
    spreadsheet_column_mapping = forms.CharField(required=False, widget=forms.HiddenInput())
    spreadsheet_path_adjustment = forms.CharField(required=False, widget=forms.HiddenInput())
    upload_relative_paths = forms.CharField(required=False, widget=forms.HiddenInput())
    ml_consent = forms.BooleanField(
        widget=forms.CheckboxInput(),
        label="I agree to the use of my uploaded images and videos for training AI models.",
        required=True,
    )

    def __init__(self, *args, **kwargs):
        user = kwargs.pop("user", None)
        super().__init__(*args, **kwargs)
        self.fields["ml_consent"].initial = user.caiduser.ml_consent_given if user else False
        self.can_use_taxon_classification = bool(user and user.caiduser.show_taxon_classification)
        self.can_use_reid = bool(user and user.caiduser.show_reid)
        self.show_upload_target_choice = self.can_use_taxon_classification and self.can_use_reid
        self.can_choose_identified_dataset = bool(
            user
            and self.can_use_reid
            and (
                user.caiduser.show_base_dataset
                or user.caiduser.show_base_between_regular_uploads
            )
            and (user.is_staff or user.caiduser.workgroup_admin)
        )
        self.show_reid_options = self.can_use_reid and (self.show_upload_target_choice or not self.can_use_taxon_classification)

        if self.show_upload_target_choice:
            self.fields["upload_target"].required = True
            self.fields["upload_target"].initial = "taxon_processing"
        elif self.can_use_reid:
            self.fields["upload_target"].initial = "identification"
            self.fields["upload_target"].widget = forms.HiddenInput()
        else:
            self.fields["upload_target"].initial = "taxon_processing"
            self.fields["upload_target"].widget = forms.HiddenInput()

        self.fields["taxon_mode"].widget = forms.HiddenInput()
        self.fields["taxon_for_identification"].widget = forms.HiddenInput()
        self.fields["is_for_identification"].widget = forms.HiddenInput()
        if not self.can_choose_identified_dataset:
            self.fields["contains_identities"].widget = forms.HiddenInput()

    def clean(self):
        cleaned_data = super().clean()
        cleaned_data["path_regex"] = normalize_regex_input(cleaned_data.get("path_regex", ""))
        upload_target = cleaned_data.get("upload_target") or self.fields["upload_target"].initial
        if not self.can_use_taxon_classification and self.can_use_reid:
            upload_target = "identification"
        elif self.can_use_taxon_classification and not self.can_use_reid:
            upload_target = "taxon_processing"
        cleaned_data["upload_target"] = upload_target
        spreadsheet_file = cleaned_data.get("spreadsheet_file")

        if self.show_upload_target_choice and not upload_target:
            self.add_error("upload_target", "Choose whether this upload should start in taxon processing or re-identification.")

        if upload_target == "identification":
            cleaned_data["is_for_identification"] = True
            cleaned_data["contains_single_taxon"] = True
            cleaned_data["taxon_mode"] = "single_taxon"
            if not self.can_choose_identified_dataset:
                cleaned_data["contains_identities"] = False
        else:
            cleaned_data["is_for_identification"] = False
            cleaned_data["contains_single_taxon"] = False
            cleaned_data["contains_identities"] = False
            cleaned_data["taxon_mode"] = "recognize_taxa"
        cleaned_data["taxon_for_identification"] = None

        if spreadsheet_file:
            suffix = Path(spreadsheet_file.name).suffix.lower()
            if suffix not in (".csv", ".xlsx"):
                self.add_error("spreadsheet_file", "Only CSV and XLSX files are supported.")

        return cleaned_data


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
            "location",
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
        self.fields["locality"].widget.attrs["class"] = (
            self.fields["locality"].widget.attrs.get("class", "") + " js-searchable-select"
        ).strip()
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
        self.fields["locality"].widget.attrs["class"] = (
            self.fields["locality"].widget.attrs.get("class", "") + " js-searchable-select"
        ).strip()
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


class MediaFileFilenameMetadataForm(forms.Form):
    directory_mapping = forms.CharField(required=False, widget=forms.HiddenInput())
    path_regex = forms.CharField(
        label="Path / filename regex",
        required=False,
        widget=forms.Textarea(attrs={"rows": 3}),
        help_text=(
            "Use named groups such as (?P<locality>...), (?P<taxon>...), "
            "(?P<unique_name>...), (?P<code>...), (?P<juv_code>...), or (?P<check_date>...). "
            "(?P<identity>...) is accepted as a legacy alias for (?P<unique_name>...)."
        ),
    )
    apply_to_manually_updated = forms.BooleanField(
        label="Apply to manually updated files",
        required=False,
        help_text="When unchecked, media files with updated_by set are skipped.",
    )
    force_rewrite_filled_data = forms.BooleanField(
        label="Force rewrite filled data",
        required=False,
        help_text="When unchecked, only empty fields are filled.",
    )

    def clean_path_regex(self):
        return normalize_regex_input(self.cleaned_data["path_regex"])


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
    code = forms.ChoiceField(choices=[], required=False)
    juv_code = forms.ChoiceField(choices=[], required=False)
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
        if self.instance and self.instance.mediafile_id:
            workgroup = self.instance.mediafile.parent.owner.workgroup
            self.fields["identity"].queryset = self.fields["identity"].queryset.filter(owner_workgroup=workgroup)

        self.fields["identity"].queryset = self.fields["identity"].queryset.order_by("name")
        self.fields["taxon"].queryset = models.Taxon.objects.order_by("name")
        self.fields["identity"].widget.attrs["class"] = (
            self.fields["identity"].widget.attrs.get("class", "") + " js-searchable-select"
        ).strip()
        self.fields["taxon"].widget.attrs["class"] = (
            self.fields["taxon"].widget.attrs.get("class", "") + " js-searchable-select"
        ).strip()


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
