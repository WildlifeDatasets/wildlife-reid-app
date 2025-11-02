import datetime
import logging
from io import BytesIO
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import pandas as pd
from django.http import HttpResponse
from django.shortcuts import redirect

logger = logging.getLogger(__name__)


def set_sort_anything_by(request, name_plural: str, sort_by: str):
    """Sort uploaded archives by."""
    request.session[f"sort_{name_plural}_by"] = sort_by
    # request.session[f"sort_{name_plural}_by"] = sort_by

    # go back to previous page
    return redirect(request.META.get("HTTP_REFERER", "/"))


def set_item_number_anything(request, name_plural: str, item_number: int):
    """Sort uploaded archives by."""
    request.session[f"item_number_{name_plural}"] = item_number
    # go back to previous page but set ?page=1
    referer = request.META.get("HTTP_REFERER", "/")
    # rozparsuj URL
    parsed = urlparse(referer)
    query_params = parse_qs(parsed.query)

    # nastav page=1
    query_params["page"] = ["1"]

    # slož zpět
    new_query = urlencode(query_params, doseq=True)
    new_url = urlunparse(parsed._replace(query=new_query))

    return redirect(new_url)
    # # find page= and remove it
    # referer = re.sub(r"page=\d+", "", referer)
    # # add page=1
    # referer += "?page=1"
    # return redirect(referer)
    #
    # # return redirect(request.META.get("HTTP_REFERER", "/"))


def get_order_by_anything(request, name_plural: str, model=None):
    """Get order by for uploaded archives."""
    direction = "desc"

    def get_session_key(param: str) -> str:
        return f"{param}_{name_plural}"

    sort = request.GET.get("sort") or request.session.get(get_session_key("sort"))
    direction = request.GET.get("dir") or request.session.get(get_session_key("dir"), direction)
    # page = request.GET.get("page") or request.session.get(get_session_key("page"), 1)
    logger.debug(f"{sort=}, {direction=}")

    if sort:
        request.session[get_session_key("sort")] = sort
    else:
        if name_plural == "uploaded_archives":
            sort = "uploaded_at"
        else:
            sort = "name"

    request.session[get_session_key("dir")] = direction
    # request.session[get_session_key("page")] = page

    # sort_by = request.session.get(f"sort_{name_plural}_by", default)
    return sort, direction


def get_item_number_anything(request, name_plural: str):
    """Get order by for uploaded archives."""
    default = 10
    item_number = request.session.get(f"item_number_{name_plural}", default)
    return item_number


def excel_response(df, name):
    """Return Excel HttpResponse from DataFrame."""
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create a BytesIO buffer to save the Excel file
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=name)
    # Rewind the buffer

    output.seek(0)

    response = HttpResponse(output, content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    response["Content-Disposition"] = f"attachment; filename={name}.{datetime_str}.xlsx"
    return response


def csv_response(df, name):
    """Return CSV HttpResponse from DataFrame."""
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    response = HttpResponse(df.to_csv(encoding="utf-8"), content_type="text/csv")
    response["Content-Disposition"] = f"attachment; filename={name}.{datetime_str}.csv"
    return response
