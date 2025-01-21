import logging
from io import BytesIO
from pathlib import Path
from typing import Union, List

import pandas as pd
from django.forms import modelformset_factory
from django.http import HttpResponseNotAllowed
from django.shortcuts import HttpResponse, get_object_or_404, redirect, render
from django.urls import reverse_lazy
from django.db.models import QuerySet
from django.contrib.auth.decorators import login_required, user_passes_test
# from torch.serialization import locality_tag
import plotly.graph_objects as go

from . import forms, model_tools
from .forms import LocalityForm
from .model_extra import (
    prepare_dataframe_for_uploads_in_one_locality,
    user_has_rw_acces_to_uploadedarchive, )
from .models import Locality, UploadedArchive, MediaFile, user_has_access_filter_params, get_all_relevant_localities
from . import models
import logging

logger = logging.getLogger("app")


def _round_location(locality: Locality, order: int = 3):
    """Round location for anonymization."""
    if (locality.location is None) or (locality.location == ""):
        return locality.location
    lat, lon = str(locality.location).split(",")
    lat = round(float(lat), order)
    lon = round(float(lon), order)
    locality.location = f"{lat},{lon}"
    locality.save()
    return f"{lat},{lon}"


def delete_locality(request, locality_id):
    """Delete locality."""
    get_object_or_404(
        Locality,
        pk=locality_id,
        **user_has_access_filter_params(request.user.caiduser, "owner"),
    ).delete()
    return redirect("caidapp:manage_localities")


def update_locality(request, locality_id=None):
    """Show and create or update location."""
    if locality_id is None:
        locality = Locality(
            owner=request.user.caiduser,
        )
    else:
        locality = get_object_or_404(
            Locality,
            pk=locality_id,
            **user_has_access_filter_params(request.user.caiduser, "owner"),
        )
    if request.method == "POST":
        form = LocalityForm(request.POST, instance=locality)
        if form.is_valid():

            # get uploaded archive
            locality = form.save()
            _round_location(locality, order=3)
            return redirect("caidapp:localities")
    else:
        form = LocalityForm(instance=locality)

    if locality_id is None:
        delete_button_url = None
    else:
        delete_button_url = reverse_lazy(
            "caidapp:delete_locality", kwargs={"locality_id": locality_id}
        )

    return render(
        request,
        "caidapp/update_form.html",
        {
            "form": form,
            "headline": "Locality",
            "button": "Save",
            "locality": locality,
            "delete_button_url": delete_button_url,
        },
    )


def manage_localities(request):
    """Add new locality or update names of localities."""
    LocalityFormSet = modelformset_factory(
        Locality, fields=("name",), can_delete=False, can_order=False
    )
    params = user_has_access_filter_params(request.user.caiduser, "owner")
    formset = LocalityFormSet(queryset=Locality.objects.filter(**params))

    if request.method == "POST":
        form = LocalityFormSet(request.POST)
        if form.is_valid():
            form.save()
    else:
        form = formset

    return render(
        request,
        "caidapp/manage_localities.html",
        {
            "page_obj": form,
        },
    )


def _set_localities_to_mediafiles_of_uploadedarchive(
    request, uploaded_archive: UploadedArchive, locality: Locality
):
    """Set locality to mediafiles of uploaded archive."""
    if not user_has_rw_acces_to_uploadedarchive(request.user.caiduser, uploaded_archive):
        return HttpResponseNotAllowed("Not allowed to edit this uploaded archive.")
    mediafiles = uploaded_archive.mediafile_set.all()
    for mediafile in mediafiles:
        mediafile.locality = locality
        mediafile.save()


def export_localities_view(request):
    """Export localities."""
    localities = Locality.objects.filter(
        **user_has_access_filter_params(request.user.caiduser, "owner")
    )
    df = pd.DataFrame.from_records(localities.values())[["name", "location"]]
    response = HttpResponse(df.to_csv(encoding="utf-8"), content_type="text/csv")
    response["Content-Disposition"] = "attachment; filename=localities.csv"
    return response


def export_localities_view_xls(request):
    """Export localities."""
    localities = Locality.objects.filter(
        **user_has_access_filter_params(request.user.caiduser, "owner")
    )
    df = pd.DataFrame.from_records(localities.values())[["name", "location"]]

    # Create a BytesIO buffer to save the Excel file
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Localities")

    # Rewind the buffer
    output.seek(0)

    response = HttpResponse(
        output, content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    response["Content-Disposition"] = "attachment; filename=localities.xlsx"
    return response


def import_localities_view(request):
    """Import localities."""
    logger.debug(f"Importing localities, method {request.method}")
    if request.method == "POST":
        form = forms.LocalityImportForm(request.POST, request.FILES)
        if form.is_valid():
            logger.debug("form is valid")
            file = form.cleaned_data["spreadsheet_file"]

            file_ext = Path(file.name).suffix.lower()
            file_content = file.read()
            rename_columns = {
                "Location": "location",
                "Latitude": "latitude",
                "Longitude": "longitude",
            }


            if file_ext == ".xlsx":
                df = pd.read_excel(BytesIO(file_content))
                df.rename(columns=rename_columns, inplace=True)
            elif file_ext == ".csv":
                df = pd.read_csv(BytesIO(file_content))
                df.rename(columns=rename_columns, inplace=True)
            else:
                return HttpResponse("Only .xlsx and .csv files are supported.")

            for index, row in df.iterrows():
                locality = models.get_locality(request.user.caiduser, row["name"])
                locality.name = row["name"]
                if "location" in df.keys():
                    locality.location = row["location"]
                elif "latitude" in df.keys() and "longitude" in df.keys():
                    locality.location = f"{row['latitude']},{row['longitude']}"
                if locality.owner is None:
                    locality.owner = request.user.caiduser
                locality.save()
            return redirect("caidapp:localities")
    else:
        form = forms.LocalityImportForm()
    return render(
        request,
        # "caidapp/model_form_upload.html",
        "caidapp/update_form.html",
        {
            "form": form,
            "headline": "Import localities",
            "button": "Import",
            "text_note": "Upload CSV or XLSX file. "
            + "There should be columns 'name' and 'location' in the file. "
            + "Location should be in format 'lat,lon'.",
            "next": "caidapp:localitys",
        },
    )


def uploads_of_locality(request, locality_hash):
    """Show all uploads of a location."""
    locality = get_object_or_404(
        Locality,
        hash=locality_hash,
        **user_has_access_filter_params(request.user.caiduser, "owner"),
    )
    uploaded_archives = locality.uploadedarchive_set.all()
    return render(
        request,
        "caidapp/uploads_location.html",
        {"location": locality, "page_obj": uploaded_archives},
    )


def download_records_from_locality_csv_view(request, locality_hash):
    """Download records from location."""
    locality = get_object_or_404(
        Locality,
        hash=locality_hash,
        **user_has_access_filter_params(request.user.caiduser, "owner"),
    )

    df = prepare_dataframe_for_uploads_in_one_locality(locality.id)
    response = HttpResponse(df.to_csv(encoding="utf-8"), content_type="text/csv")
    response["Content-Disposition"] = "attachment; filename=location_checks.csv"
    return response


def download_records_from_locality_xls_view(request, locality_hash):
    """Download records from location."""
    location = get_object_or_404(
        Locality,
        hash=locality_hash,
        **user_has_access_filter_params(request.user.caiduser, "owner"),
    )

    df = prepare_dataframe_for_uploads_in_one_locality(location.id)
    df = model_tools.convert_datetime_to_naive(df)

    # Create a BytesIO buffer to save the Excel file
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Location Checks")

    # Rewind the buffer
    output.seek(0)

    response = HttpResponse(
        output, content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    response["Content-Disposition"] = "attachment; filename=location_checks.xlsx"
    return response


def create_map_from_mediafiles(mediafiles: Union[QuerySet, List[MediaFile]]):
    """Create dataframe from mediafiles."""
    # create dataframe
    logger.debug(f"creating map for media files: {mediafiles=}")

    queryset_list = list(mediafiles.values("id", "locality__name", "locality__location"))
    df = pd.DataFrame.from_records(queryset_list)
    logger.debug(f"{list(df.keys())}")
    logger.debug(f"{df.shape=}")
    data = []
    for mediafile in mediafiles:
        if (
                mediafile.locality
                and mediafile.locality.location
                and mediafile.locality.location.count(",") == 1
        ):
            row = {
                "id": mediafile.id,
                "category": mediafile.category.name if mediafile.category else None,
                "category_id": mediafile.category.id if mediafile.category else None,
                'captured_at': mediafile.captured_at if mediafile.captured_at else None,
                "locality": mediafile.locality.name if mediafile.locality else None,
                "locality__location": mediafile.locality.location
                if mediafile.locality.location
                else None,
            }
            data.append(row)

    df2 = pd.DataFrame.from_records(data)
    if "locality__location" not in df2.keys():
        return None
    df2[["lat", "lon"]] = df2["locality__location"].str.split(",", expand=True)
    df2["lat"] = df2["lat"].astype(float)
    df2["lon"] = df2["lon"].astype(float)
    df2 = df2.sort_values("captured_at")

    # Assign colors based on sequence
    df2["color"] = pd.cut(
        df2.index, bins=len(df2), labels=range(len(df2)), include_lowest=True
    )
    step_r = 255 / len(df2)
    step_g = 155 / len(df2)
    step_b = 105 / len(df2)
    # color_scale = [f"rgba({int(255 - i*step_r)}, {int(100 + i*step_g)}, {int(150 + i*step_b)}, 160)" for i in range(len(df2))]
    color_scale = [f"rgba(10,100,100, 160)" for i in range(len(df2))]



    logger.debug(f"{list(df2.keys())}")
    # if len(df2) > 10:
    #     logger.debug(f"{df2.sample(10).to_dict()=}")
    # else:
    #     logger.debug(f"{df2.to_dict()=}")

    # Calculate the range of your data to set the zoom level
    lat_range = df2["lat"].max() - df2["lat"].min()
    lon_range = df2["lon"].max() - df2["lon"].min()

    # Set an appropriate zoom level based on the maximum range
    max_range = max(lat_range, lon_range)
    zoom = 0  # Set a default zoom level
    if max_range < 10:
        zoom = 6
    elif max_range < 30:
        zoom = 5
    elif max_range < 60:
        zoom = 4
    else:
        zoom = 3  # For larger ranges, set a smaller zoom level

    # fig = go.Figure(go.Densitymapbox(lat=df2.lat, lon=df2.lon, radius=10, showscale=False))
    # fig.update_layout(
    #     mapbox_style="open-street-map",
    #     mapbox_center_lon=df2.lon.unique().mean(),
    #     mapbox_center_lat=df2.lat.unique().mean(),
    #     mapbox_zoom=zoom,
    # )
    #
    # fig.update_layout(margin={"r": 0, "t": 10, "l": 0, "b": 0}, height=300)

    # Create the base map
    fig = go.Figure()

    # Add a density map for points
    fig.add_trace(
        go.Densitymapbox(
            lat=df2.lat,
            lon=df2.lon,
            radius=10,
            showscale=False,
            name = 'Density Map',
        )
    )

    # Add a Scattermapbox trace to connect points with lines
    fig.add_trace(
        go.Scattermapbox(
            lat=df2.lat,
            lon=df2.lon,
            mode='lines+markers',  # Shows both lines and markers
            marker=dict(size=7),  # Adjust marker size
            # line=dict(width=2, color=[color_scale[i] for i in range(len(df2))]),  # Adjust line style
            line=dict(width=2, color="blue"),  # Adjust line style
            name = 'Path',  # Legend label
            visible = True  # Default visibility
        )
    )

    # Add arrow for last segment
    if len(df2) > 1:
        fig.add_trace(
            go.Scattermapbox(
                lat=df2.lat.iloc[-2:],  # Last two points
                lon=df2.lon.iloc[-2:],
                mode='lines+markers',
                line=dict(width=3, color='red'),  # Arrow color
                marker=dict(size=10, symbol='arrow-bar'),  # Arrowhead
                name='Last Segment',
            )
        )

    # Update layout with map style and center
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center_lon=df2.lon.mean(),
        mapbox_center_lat=df2.lat.mean(),
        mapbox_zoom=zoom,
        margin={"r": 0, "t": 10, "l": 0, "b": 30},
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",  # Horizontal orientation
            yanchor="top",  # Align to the top of the legend
            y=-0.1,  # Place below the map (negative value for outside the map area)
            xanchor="center",  # Center align horizontally
            x=0.5  # Position at the center
        )

    )

    map_html = fig.to_html()
    return map_html


@login_required
def localities_view(request):
    """List of localities."""
    localities = get_all_relevant_localities(request)
    logger.debug(f"{len(localities)=}")
    return render(request, "caidapp/localities.html", {"localities": localities})
