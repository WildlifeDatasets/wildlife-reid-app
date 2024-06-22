from django.shortcuts import Http404, HttpResponse, get_object_or_404, redirect, render
from django.http import StreamingHttpResponse, Http404
from django.shortcuts import get_object_or_404
from .models import MediaFile
import os

def stream_video(request, mediafile_id):
    mediafile = get_object_or_404(MediaFile, id=mediafile_id)
    if mediafile.media_type != "video":
        raise Http404("Not a video file")

    video_path = mediafile.mediafile.path
    if not os.path.exists(video_path):
        raise Http404()

    def file_iterator(file_name, chunk_size=8192):
        with open(file_name, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    response = StreamingHttpResponse(file_iterator(video_path), content_type='video/mp4')
    response['Content-Length'] = os.path.getsize(video_path)
    response['Accept-Ranges'] = 'bytes'

    return response
