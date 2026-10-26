import re
from pathlib import Path
from tempfile import TemporaryDirectory

from django.conf import settings
from django.core.management import call_command
from django.template import Context, Template
from django.test import SimpleTestCase


class BaseTemplateStaticAssetsTest(SimpleTestCase):
    def test_base_template_assets_render_with_production_manifest(self):
        """Collect real app assets and resolve the base template's static tags with DEBUG disabled."""
        app_dir = Path(settings.BASE_DIR) / "caidapp"
        source = (app_dir / "templates" / "base.html").read_text(encoding="utf8")
        tags = re.findall(r"{%\s*static\s+['\"][^'\"]+['\"]\s*%}", source)
        self.assertTrue(tags)
        with TemporaryDirectory() as static_root, self.settings(
            DEBUG=False,
            STATIC_ROOT=static_root,
            STATICFILES_DIRS=[app_dir / "static"],
            STATICFILES_FINDERS=["django.contrib.staticfiles.finders.FileSystemFinder"],
            STORAGES={
                "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
                "staticfiles": {"BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage"},
            },
        ):
            call_command("collectstatic", interactive=False, verbosity=0)
            rendered = Template("{% load static %}" + "\n".join(tags)).render(Context())
            self.assertNotIn("{%", rendered)
            self.assertTrue((Path(static_root) / "staticfiles.json").is_file())
