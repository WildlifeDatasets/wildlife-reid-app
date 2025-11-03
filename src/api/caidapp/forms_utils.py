class ModelHelpTextMixin:
    """Add help_text from model to form fields."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model = getattr(self._meta, "model", None)
        if not model:
            return

        for name, field in self.fields.items():
            if not field.help_text:
                try:
                    model_field = model._meta.get_field(name)
                    if model_field.help_text:
                        field.help_text = model_field.help_text
                except Exception:
                    pass
