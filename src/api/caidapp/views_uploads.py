from django.shortcuts import render
from django.contrib import messages
from typing import Optional
from . import views
from .models import UploadedArchive
import logging
from django.shortcuts import redirect
from django.http import HttpResponse
from django.utils import timezone


logger = logging.getLogger(__name__)


def taxon_processing(request):

    btn_styles, btn_tooltips = views._multiple_species_button_style_and_tooltips(request)
    return render(
        request,
        # "caidapp/uploads_species.html",
        "caidapp/taxon_processing.html",
        {
            "btn_styles": btn_styles, "btn_tooltips": btn_tooltips},
    )
    pass


from collections import defaultdict


def _get_check_dates(request, contains_single_taxon: Optional[bool] = None,
                                 taxon_for_identification__isnull: Optional[bool] = None

) -> list:
    """Get the list of unique dates for the uploaded archives."""

    filter_params = {}
    if contains_single_taxon is not None:
        filter_params = dict(contains_single_taxon=contains_single_taxon)
    elif taxon_for_identification__isnull is not None:
        filter_params = dict(taxon_for_identification__isnull=taxon_for_identification__isnull)

    uploaded_archives = UploadedArchive.objects.filter(
        **views.get_content_owner_filter_params(request.user.caiduser, "owner"),
        **filter_params
    )

    # get the list of unique dates
    dates = uploaded_archives.values_list("location_check_at", flat=True).distinct()
    # logger.debug(f"dates: {dates}")
    # messages.info(request, f"{dates=}")

    # get date from DateTimeField
    dates = [timezone.localtime(date).date() for date in dates if date is not None]
    # messages.info(request, f"{dates=}")
    # logger.debug(f"dates: {dates}")
    # get list of years
    # Organize into a structure by year, month, and sorted
    return dates


def _get_grouped_dates(dates) -> dict:
    """Group dates together by year and month."""
    grouped_dates = defaultdict(lambda: defaultdict(list))
    for date in dates:
        year = date.year
        month = date.strftime('%B')
        grouped_dates[year][month].append(date)

    # Sort years, months, and dates in descending order
    sorted_grouped_dates = dict(sorted(grouped_dates.items(), reverse=True))
    for year in sorted_grouped_dates:
        sorted_grouped_dates[year] = dict(sorted(sorted_grouped_dates[year].items(), reverse=True))
        for month in sorted_grouped_dates[year]:
            sorted_grouped_dates[year][month].sort(reverse=True)

    return sorted_grouped_dates

def camera_trap_check_dates_view(request, contains_single_taxon: Optional[bool] = None,
                                 taxon_for_identification__isnull: Optional[bool] = None, year: Optional[int] = None,):

        #
    dates = _get_check_dates(request, contains_single_taxon, taxon_for_identification__isnull)
    sorted_grouped_dates = _get_grouped_dates(dates)

    # get list of years
    years = list(sorted_grouped_dates.keys())

    if (year is None) and (len(years) > 0):
        year = years[0]
    # select just one year
    if year is not None:
        sorted_grouped_dates = {year: sorted_grouped_dates[year]}

    # messages.info(request, f"Number of dates: {len(dates)}")
    # messages.info(request, f"{sorted_grouped_dates=}")

    return render(
        request,
        "caidapp/check_dates.html", context={
            "grouped_dates": sorted_grouped_dates,
            "years": years,
            "selected_year": year
        }
    )



def camera_trap_check_date_view(request, date: Optional[str] = None, contains_single_taxon: Optional[bool] = None,
                                 taxon_for_identification__isnull: Optional[bool] = None) -> HttpResponse:

    from .views import _uploads_general

    # this is there only to get list o years. Maybe it is no necessary
    dates = _get_check_dates(request, contains_single_taxon, taxon_for_identification__isnull)
    sorted_grouped_dates = _get_grouped_dates(dates)
    # get list of years
    years = list(sorted_grouped_dates.keys())

    if date is None:
        filter = dict(location_check_at__isnull=True)
        date = "None"
    else:
        filter = dict(location_check_at__date=date)
    page_context = _uploads_general(
        request, contains_single_taxon=contains_single_taxon,
        taxon_for_identification__isnull=taxon_for_identification__isnull,
        **filter,
    )

    return render(
        request,
        "caidapp/uploads_species.html",
        context={
            **page_context,
            "date": date,
            "years":years,
            # "page_obj": page_obj,
            # "elided_page_range": elided_page_range,
            # "btn_styles": _single_species_button_style(request),
        },
    )


def uploadedarchive_detail(request, uploadedarchive_id: int) -> HttpResponse:
    from .views import _uploads_general
    uploaded_archive = UploadedArchive.objects.get(id=uploadedarchive_id)

    # {% if uploadedarchive.count_of_mediafiles_with_taxon_for_identification %}
    # Mediafiles with taxon {{ uploadedarchive.taxon_for_identification }}: {{ uploadedarchive.count_of_mediafiles_with_taxon_for_identification }}
    # {% endif %}
    #
    # {% if uploadedarchive.animal_number %}
    # Count: {{ uploadedarchive.animal_number }}
    dictionary = {
        "Uploaded at": uploaded_archive.uploaded_at,
        "Location": uploaded_archive.location_at_upload_object,
        "Location check at": uploaded_archive.location_check_at,
        "Status": uploaded_archive.taxon_status,
        "Status message": uploaded_archive.status_message,
        "Identification status": uploaded_archive.identification_status,
        "Owner": uploaded_archive.owner,
        "Contains single taxon": uploaded_archive.contains_single_taxon,
        "Taxon for identification": uploaded_archive.taxon_for_identification,
        "Count of media files": uploaded_archive.count_of_mediafiles(),
        "Count of representative media files": uploaded_archive.count_of_representative_mediafiles(),
        "Count of media files with taxon": uploaded_archive.count_of_mediafiles_with_taxon(),
        "Count of media files withi missing taxon": uploaded_archive.count_of_mediafiles_with_missing_taxon(),
        "Count of media files with verified taxon": uploaded_archive.count_of_mediafiles_with_verified_taxon(),
        "Count of taxons":  uploaded_archive.count_of_taxons(),
        "Count of identities": uploaded_archive.count_of_identities(),
        "Precents of media files with taxon": uploaded_archive.percents_of_mediafiles_with_taxon(),

    }
    return render(
        request,
        "caidapp/message.html",
        context=dict(
            headline=f"Uploaded Archive {uploaded_archive.location_at_upload_object} {uploaded_archive.location_check_at}",
            dictionary=dictionary
        )
    )

