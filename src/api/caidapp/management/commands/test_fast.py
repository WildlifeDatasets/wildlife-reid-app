from django.core.management.commands.test import Command as DjangoTestCommand


class Command(DjangoTestCommand):
    help = "Run Django tests while excluding tests tagged with 'long' by default."

    def handle(self, *test_labels, **options):
        if not options.get("tag"):
            excluded_tags = list(options.get("exclude_tag") or [])
            if "long" not in excluded_tags:
                excluded_tags.append("long")
            options["exclude_tag"] = excluded_tags
        return super().handle(*test_labels, **options)
