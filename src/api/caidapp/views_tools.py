



def add_querystring_to_context(request, context):
    """
    Add the query string to the context so it can be used in the template for pagination.
    """
    query_params = request.GET.copy()
    query_params.pop('page', None)
    context['query_string'] = query_params.urlencode()
    return context
