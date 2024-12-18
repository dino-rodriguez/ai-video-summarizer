import os

import click
from dotenv import load_dotenv
from openai import OpenAI

from lib.audio_tools import generate_markdown, summarize_transcript, transcribe_audio
from lib.yt import download_audio


@click.option("--url", type=str, required=False)
@click.option("--audio-path", type=str, required=False)
@click.option("--transcript-path", type=str, required=False)
@click.option("--summaries-path", type=str, required=False)
@click.command()
def main(
    url: str | None,
    audio_path: str | None,
    transcript_path: str | None,
    summaries_path: str | None,
):
    """Command line tool to generate lecture notes from YouTube video"""

    load_dotenv()
    open_ai_client = OpenAI()

    DOWNLOADS_FOLDER = "generated/downloads"
    TRANSCRIPTS_FOLDER = "generated/transcripts"
    SUMMARIES_FOLDER = "generated/summaries"
    NOTES_FOLDER = "generated/notes"

    if url is not None:
        print(f"Downloading audio from URL: {url}")
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

    if summaries_path is not None:
        notes_filename = summaries_path.split("/")[-1].rstrip(".txt") + ".md"
        notes_path = os.path.join(NOTES_FOLDER, notes_filename)
        os.makedirs(NOTES_FOLDER, exist_ok=True)
        generate_markdown(summaries_path, notes_path)


if __name__ == "__main__":
    main()
