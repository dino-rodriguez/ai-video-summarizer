import os

import click
from dotenv import load_dotenv
from openai import OpenAI

from lib.audio_tools import summarize_transcript, transcribe_audio
from lib.video_tools import extract_key_frames, save_frames
from lib.yt import download_audio, download_video


@click.option("--url", type=str, required=False)
@click.option("--video-path", type=str, required=False)
@click.option("--audio-path", type=str, required=False)
@click.option("--transcript-path", type=str, required=False)
@click.command()
def main(
    url: str | None,
    video_path: str | None,
    audio_path: str | None,
    transcript_path: str | None,
):
    """Command line tool to generate lecture notes from YouTube video"""

    load_dotenv()
    open_ai_client = OpenAI()

    DOWNLOADS_FOLDER = "generated/downloads"
    TRANSCRIPTS_FOLDER = "generated/transcripts"
    SUMMARIES_FOLDER = "generated/summaries"
    FRAMES_FOLDER = "generated/frames"

    if url is not None:
        print(f"Downloading video and audio from URL: {url}")
        video_path = download_video(url, DOWNLOADS_FOLDER)
        audio_path = download_audio(url, DOWNLOADS_FOLDER)

    if audio_path is not None:
        if not audio_path.endswith(".m4a"):
            raise Exception("Audio file must be m4a")

        print(f"Transcribing audio file: {audio_path}")

        transcript_filename = audio_path.split("/")[-1].rstrip(".audio.m4a") + ".txt"
        transcript_path = os.path.join(TRANSCRIPTS_FOLDER, transcript_filename)
        os.makedirs(TRANSCRIPTS_FOLDER, exist_ok=True)
        transcribe_audio(audio_path, transcript_path)

        print(f"Generated transcript: {transcript_path}")

    if transcript_path is not None:
        print(f"Generating summaries from transcript: {transcript_path}")

        summaries_filename = transcript_path.split("/")[-1]
        summaries_path = os.path.join(SUMMARIES_FOLDER, summaries_filename)
        os.makedirs(SUMMARIES_FOLDER, exist_ok=True)
        summarize_transcript(open_ai_client, transcript_path, summaries_path)

    if video_path is not None:
        if not video_path.endswith(".mp4"):
            raise Exception("Video file must be mp4")

        print(f"Extracting key frames from video: {video_path}")
        key_frames = extract_key_frames(video_path)
        frames_dirname = video_path.split("/")[-1].rstrip(".video.mp4")
        frames_path = os.path.join(FRAMES_FOLDER, frames_dirname)
        save_frames(key_frames, frames_path)
        print(f"Saved {len(key_frames)} frames to {frames_path}")


if __name__ == "__main__":
    main()
