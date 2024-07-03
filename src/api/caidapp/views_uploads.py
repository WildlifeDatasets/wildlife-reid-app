


def give_me_taxon_on_unknown_mediafiles(request):
    """
    This view is used to display a list of mediafiles that have not yet been identified.
    """
    filter
    mediafiles = MediaFile.objects.filter(taxon=None)
    return render(request, "caidapp/mediafiles.html", {"mediafiles": mediafiles})