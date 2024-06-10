from io import BytesIO
from pathlib import Path
import logging

import pandas as pd

from django.shortcuts import Http404, HttpResponse, get_object_or_404, redirect, render
from django.http import HttpResponseBadRequest, HttpResponseNotAllowed, JsonResponse
from django.forms import modelformset_factory
from django.urls import reverse_lazy

from . import forms
from .forms import LocationForm
from .models import Location, get_content_owner_filter_params, UploadedArchive
from .model_extra import _user_has_rw_acces_to_uploadedarchive

logger = logging.getLogger("app")

def _round_location(location: Location, order: int = 3):
    """Round location for anonymization."""
    if (location.location is None) or (location.location == ""):
        return location.location
    lat, lon = str(location.location).split(",")
    lat = round(float(lat), order)
    lon = round(float(lon), order)
    location.location = f"{lat},{lon}"
    location.save()
    return f"{lat},{lon}"


def delete_location(request, location_id):
    """Delete location."""
    get_object_or_404(
        Location,
        pk=location_id,
        **get_content_owner_filter_params(request.user.caiduser, "owner"),
    ).delete()
    return redirect("caidapp:manage_locations")


def update_location(request, location_id):
    """Show and update location."""
    location = get_object_or_404(
        Location,
        pk=location_id,
        **get_content_owner_filter_params(request.user.caiduser, "owner"),
    )
    if request.method == "POST":
        form = LocationForm(request.POST, instance=location)
        if form.is_valid():

            # get uploaded archive
            location = form.save()
            _round_location(location, order=3)
            return redirect("caidapp:locations")
    else:
        form = LocationForm(instance=location)
    return render(
        request,
        "caidapp/update_form.html",
        {"form": form, "headline": "Location", "button": "Save", "location": location,
         "delete_button_url": reverse_lazy('caidapp:delete_location', kwargs={'location_id': location_id}),
         },
    )


def manage_locations(request):
    """Add new location or update names of locations."""
    LocationFormSet = modelformset_factory(
        Location, fields=("name",), can_delete=False, can_order=False
    )
    params = get_content_owner_filter_params(request.user.caiduser, "owner")
    formset = LocationFormSet(queryset=Location.objects.filter(**params))

    if request.method == "POST":
        form = LocationFormSet(request.POST)
        if form.is_valid():
            form.save()
    else:
        form = formset

    return render(
        request,
        "caidapp/manage_locations.html",
        {
            "page_obj": form,
        },
    )


def _get_all_user_locations(request):
    """Get all users locations."""
    params = get_content_owner_filter_params(request.user.caiduser, "owner")
    # logger.debug(f"{params=}")
    locations = Location.objects.filter(**params).order_by("name")
    return locations


def _set_location_to_mediafiles_of_uploadedarchive(
    request, uploaded_archive: UploadedArchive, location: Location
):
    """Set location to mediafiles of uploaded archive."""
    if not _user_has_rw_acces_to_uploadedarchive(request.user.caiduser, uploaded_archive):
        return HttpResponseNotAllowed("Not allowed to edit this uploaded archive.")
    mediafiles = uploaded_archive.mediafile_set.all()
    for mediafile in mediafiles:
        mediafile.location = location
        mediafile.save()


def export_locations_view(request):
    """Export locations."""

    locations = Location.objects.filter(
        **get_content_owner_filter_params(request.user.caiduser, "owner")
    )
    df = pd.DataFrame.from_records(locations.values())[["name", "location"]]
    response = HttpResponse(df.to_csv(encoding="utf-8"), content_type="text/csv")
    response["Content-Disposition"] = "attachment; filename=locations.csv"
    return response


def export_locations_view_xls(request):
    """Export locations."""

    locations = Location.objects.filter(
        **get_content_owner_filter_params(request.user.caiduser, "owner")
    )
    df = pd.DataFrame.from_records(locations.values())[["name", "location"]]

    # Create a BytesIO buffer to save the Excel file
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Locations')

    # Rewind the buffer
    output.seek(0)

    response = HttpResponse(output,
                            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=locations.xlsx'
    return response


def import_locations_view(request):
    """Import locations."""
    logger.debug(f"Importing locations, method {request.method}")
    if request.method == "POST":
        form = forms.LocationImportForm(request.POST, request.FILES)
        if form.is_valid():
            logger.debug("form is valid")
            file = form.cleaned_data["spreadsheet_file"]

            file_ext = Path(file.name).suffix.lower()
            file_content = file.read()

            if file_ext == ".xlsx":
                df = pd.read_excel(BytesIO(file_content))
            elif file_ext == ".csv":
                df = pd.read_csv(BytesIO(file_content))
            else:
                return HttpResponse("Only .xlsx and .csv files are supported.")

            for index, row in df.iterrows():
                location = Location()
                location.name = row["name"]
                location.location = row["location"]
                location.owner = request.user.caiduser
                location.save()
            return redirect("caidapp:locations")
    else:
        form = forms.LocationImportForm()
    return render(
        request,
        # "caidapp/model_form_upload.html",
        "caidapp/update_form.html",
        {
            "form": form,
            "headline": "Import locations",
            "button": "Import",
            "text_note": "Upload CSV or XLSX file. There should be columns 'name' and 'location' in the file. Location should be in format 'lat,lon'.",
            "next": "caidapp:locations",

        },
    )
