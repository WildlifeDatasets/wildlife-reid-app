from pathlib import Path
from .models import UploadedArchive
import shutil
import os.path
import django
from django.conf import settings
from .cv import dataset_tools, data_processing_pipeline
import loguru
from loguru import logger

def run_processing_test(uploaded_archive: UploadedArchive):
    outputdir = Path(uploaded_archive.outputdir)
    if outputdir.exists() and outputdir.is_dir():
        shutil.rmtree(outputdir, ignore_errors=True)
    outputdir.mkdir(parents=True, exist_ok=True)
    log_format = loguru._defaults.LOGURU_FORMAT
    logger_id = logger.add(
        str(Path(uploaded_archive.outputdir) / "log.txt"),
        format=log_format,
        level="DEBUG",
        rotation="1 week",
        backtrace=True,
        diagnose=True,
    )
    logger.debug("Processing finished")
    logger.remove(logger_id)

def run_processing(uploaded_archive: UploadedArchive):
    outputdir = Path(uploaded_archive.outputdir)
    if outputdir.exists() and outputdir.is_dir():
        shutil.rmtree(outputdir, ignore_errors=True)
    outputdir.mkdir(parents=True, exist_ok=True)
    log_format = loguru._defaults.LOGURU_FORMAT
    logger_id = logger.add(
        str(Path(uploaded_archive.outputdir) / "log.txt"),
        format=log_format,
        level="DEBUG",
        rotation="1 week",
        backtrace=True,
        diagnose=True,
    )
    print(" processing běží")
    # logger.debug(f"Image processing of '{serverfile.mediafile}' initiated")
    # make_preview(serverfile)
    if uploaded_archive.zip_file and Path(uploaded_archive.zip_file.path).exists():
        uploaded_archive.zip_file.delete()
    input_file = Path(uploaded_archive.archivefile.path)
    # logger.debug(f"input_file={input_file}")
    outputdir = Path(uploaded_archive.outputdir)
    outputdir_images = outputdir / 'images'
    outputdir_csv = outputdir / "metadata.csv"
    outputdir_zip = outputdir / "images.zip"
    # logger.debug(f"outputdir={outputdir}")

    # _run_media_processing_rest_api(input_file, outputdir, hostname, port)

    # (outputdir / "empty.txt").touch(exist_ok=True)

    if input_file.suffix in (".mp4", ".avi"):
        pass
        # make_images_from_video(input_file, outputdir=outputdir, n_frames=1)

    data_processing_pipeline.data_processing(
        input_file, outputdir_images, outputdir_csv, num_cores=1
    )
    dataset_tools.make_zipfile(outputdir_zip, outputdir_images)

    # for video_pth in outputdir.glob("*.avi"):
    #     input_video_file = video_pth
    #     output_video_file = video_pth.with_suffix(".mp4")
    #     logger.debug(f"input_video_file={input_video_file}")
    #     logger.debug(f"outout_video_file={output_video_file}")
    #     if output_video_file.exists():
    #         output_video_file.unlink()
    #     _convert_avi_to_mp4(str(input_video_file), str(output_video_file))
    # add_generated_images(uploaded_archive)
    # make_zip(uploaded_archive)
    #
    uploaded_archive.finished_at = django.utils.timezone.now()
    uploaded_archive.zip_file = os.path.relpath(outputdir_zip, settings.MEDIA_ROOT)
    uploaded_archive.csv_file = os.path.relpath(outputdir_csv, settings.MEDIA_ROOT)
    uploaded_archive.save()
    # _add_row_to_spreadsheet(uploaded_archive, absolute_uri)
    logger.debug("Processing finished")
    logger.remove(logger_id)
