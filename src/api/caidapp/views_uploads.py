import logging
from typing import Optional

from django.http import HttpResponse
from django.shortcuts import render
from django.utils import timezone

from . import views
from .models import UploadedArchive

logger = logging.getLogger(__name__)


def taxon_processing(request):
    """View for overall management of taxon processing."""
    btn_styles, btn_tooltips = views._multiple_species_button_style_and_tooltips(request)
    return render(
        request,
        # "caidapp/uploads_species.html",
        "caidapp/taxon_processing.html",
        {"btn_styles": btn_styles, "btn_tooltips": btn_tooltips},
    )
    pass


from collections import defaultdict


def _get_check_dates(
    request,
    contains_single_taxon: Optional[bool] = None,
    taxon_for_identification__isnull: Optional[bool] = None,
) -> list:
    """Get the list of unique dates for the uploaded archives."""
    filter_params = {}
    if contains_single_taxon is not None:
        filter_params = dict(contains_single_taxon=contains_single_taxon)
    elif taxon_for_identification__isnull is not None:
        filter_params = dict(taxon_for_identification__isnull=taxon_for_identification__isnull)

    uploaded_archives = UploadedArchive.objects.filter(
        **views.get_content_owner_filter_params(request.user.caiduser, "owner"), **filter_params
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
        month = date.strftime("%B")
        grouped_dates[year][month].append(date)

    # Sort years, months, and dates in descending order
    sorted_grouped_dates = dict(sorted(grouped_dates.items(), reverse=True))
    for year in sorted_grouped_dates:
        sorted_grouped_dates[year] = dict(sorted(sorted_grouped_dates[year].items(), reverse=True))
        for month in sorted_grouped_dates[year]:
            sorted_grouped_dates[year][month].sort(reverse=True)

    return sorted_grouped_dates


def camera_trap_check_dates_view(
    request,
    contains_single_taxon: Optional[bool] = None,
    taxon_for_identification__isnull: Optional[bool] = None,
    year: Optional[int] = None,
):
    """View for checking the dates of camera trap checks."""
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
        "caidapp/check_dates.html",
        context={"grouped_dates": sorted_grouped_dates, "years": years, "selected_year": year},
    )


def camera_trap_check_date_view(
    request,
    date: Optional[str] = None,
    contains_single_taxon: Optional[bool] = None,
    taxon_for_identification__isnull: Optional[bool] = None,
) -> HttpResponse:
    """View for checking one particular date from camera checks."""
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
    page_context = views._uploads_general(
        request,
        contains_single_taxon=contains_single_taxon,
        taxon_for_identification__isnull=taxon_for_identification__isnull,
        **filter,
    )

    return render(
        request,
        "caidapp/uploads_species.html",
        context={
            **page_context,
            "date": date,
            "years": years,
            # "page_obj": page_obj,
            # "elided_page_range": elided_page_range,
            # "btn_styles": _single_species_button_style(request),
        },
    )


def uploadedarchive_detail(request, uploadedarchive_id: int) -> HttpResponse:
    """View for the detail of the uploaded archive."""
    uarch = UploadedArchive.objects.get(id=uploadedarchive_id)

    # fmt: off
    dictionary = {
        "Uploaded at": uarch.uploaded_at,
        "Location": uarch.location_at_upload_object,
        "Location check at": uarch.location_check_at,
        "Status": uarch.taxon_status,
        "Status message": uarch.status_message,
        "Identification status": uarch.identification_status,
        "Owner": uarch.owner,
        "Contains single taxon": uarch.contains_single_taxon,
        "Taxon for identification": uarch.taxon_for_identification,
        "Count of media files": uarch.count_of_mediafiles(),
        "Count of representative media files":
            uarch.count_of_representative_mediafiles(),
        "Count of media files with taxon": uarch.count_of_mediafiles_with_taxon(),
        "Count of media files with missing taxon":
            uarch.count_of_mediafiles_with_missing_taxon(),
        "Count of media files with verified taxon":
            uarch.count_of_mediafiles_with_verified_taxon(),
        "Count of taxons": uarch.count_of_taxons(),
        "Count of identities": uarch.count_of_identities(),
        "Precents of media files with taxon": uarch.percents_of_mediafiles_with_taxon(),
    }
    # fmt: on
    return render(
        request,
        "caidapp/message.html",
        context=dict(
            headline="Uploaded Archive "
            + f"{uarch.location_at_upload_object} "
            + f"{uarch.location_check_at}",
            dictionary=dictionary,
        ),
    )
