import re
from typing import Any, Literal

import yt_dlp


def download_audio(url: str, output_folder: str):
    """Download audio-only in M4A format"""

    ydl_opts_audio = {
        "outtmpl": f"./{output_folder}/%(title)s.audio.%(ext)s",  # Save audio file with .audio extension
        "format": "bestaudio[ext=m4a]",  # Download best audio-only
        "noplaylist": True,  # Single video only
    }

    return _download_yt(url, ydl_opts_audio, output_folder, "audio")


def download_video(url: str, output_folder: str):
    """Download video-only in MP4 format (up to 720p)"""

    ydl_opts_video = {
        "outtmpl": f"./{output_folder}/%(title)s.video.%(ext)s",  # Save video file with .video extension
        "format": "bestvideo[ext=mp4][height<=720]",  # Download best video-only (up to 720p)
        "noplaylist": True,  # Single video only
    }

    return _download_yt(url, ydl_opts_video, output_folder, "video")


def _download_yt(
    url: str,
    opts: dict[Any, Any],
    output_folder: str,
    media_type: Literal["video"] | Literal["audio"],
):
    """Download video or audio from YouTube"""

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if not info:
            raise Exception("Can't get video info")

        # Swap original title for kebab case
        original_title = info.get("title", "video")
        kebab_case_title = _to_kebab_case(original_title)

        # Update the output template with the kebab-case title
        opts["outtmpl"] = f"./{output_folder}/{kebab_case_title}.{media_type}.%(ext)s"

    with yt_dlp.YoutubeDL(opts) as ydl:
        # Download the video
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)
        return path


def _to_kebab_case(text: str):
    "Convert text to kebab case" ""

    words = re.split(r"[^a-zA-Z0-9]+", text)
    return "-".join(word.lower() for word in words if word)
