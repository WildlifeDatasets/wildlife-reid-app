from django.shortcuts import render
from . import views


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

