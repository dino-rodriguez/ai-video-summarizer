import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

import click
import mlx_whisper
import yt_dlp
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


@dataclass
class Chunk:
    """Describes a chunk of text from a transcript"""

    chunk: str
    prior_chunk_context: str
    word_count: int
    start_time: float
    end_time: float


@dataclass
class SummaryChunk:
    """Describes a generated summary chunk"""

    summary_chunk: str
    transcript_chunk: str
    start_time: float
    end_time: float


def to_kebab_case(text: str):
    "Convert text to kebab case" ""

    words = re.split(r"[^a-zA-Z0-9]+", text)
    return "-".join(word.lower() for word in words if word)


def download_yt(
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
        kebab_case_title = to_kebab_case(original_title)

        # Update the output template with the kebab-case title
        opts["outtmpl"] = f"./{output_folder}/{kebab_case_title}.{media_type}.%(ext)s"

    with yt_dlp.YoutubeDL(opts) as ydl:
        # Download the video
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)
        return path


def download_video(url: str, output_folder: str):
    """Download video-only in MP4 format (up to 720p)"""

    ydl_opts_video = {
        "outtmpl": f"./{output_folder}/%(title)s.video.%(ext)s",  # Save video file with .video extension
        "format": "bestvideo[ext=mp4][height<=720]",  # Download best video-only (up to 720p)
        "noplaylist": True,  # Single video only
    }

    return download_yt(url, ydl_opts_video, output_folder, "video")


def download_audio(url: str, output_folder: str):
    """Download audio-only in M4A format"""

    ydl_opts_audio = {
        "outtmpl": f"./{output_folder}/%(title)s.audio.%(ext)s",  # Save audio file with .audio extension
        "format": "bestaudio[ext=m4a]",  # Download best audio-only
        "noplaylist": True,  # Single video only
    }

    return download_yt(url, ydl_opts_audio, output_folder, "audio")


def transcribe_audio(audio_filepath: str, output_path: str):
    """Transcribe audio file to text using OpenAI Whisper"""

    result = mlx_whisper.transcribe(
        audio_filepath, path_or_hf_repo="mlx-community/whisper-turbo", verbose=False
    )

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)


def chunk_transcript(transcript_filepath: str):
    """Chunk a transcript file"""

    with open(transcript_filepath, "r") as f:
        transcript = json.load(f)

    CHUNK_LENGTH = 500
    PRIOR_CHUNK_CONTEXT_LENGTH = 200

    current_chunk = ""
    prior_chunk_context = ""
    current_chunk_word_count = 0
    previous_segment_complete_sentence = False
    current_chunk_start_time = 0.0
    current_chunk_end_time = 0.0
    chunks: list[Chunk] = []

    for segment in transcript["segments"]:
        # when we are above the chunk length and just finished a sentence,
        # append and reset to new chunk
        if (
            current_chunk_word_count >= CHUNK_LENGTH
            and previous_segment_complete_sentence
        ):
            chunks.append(
                Chunk(
                    chunk=current_chunk,
                    prior_chunk_context=prior_chunk_context,
                    word_count=current_chunk_word_count,
                    start_time=current_chunk_start_time,
                    end_time=current_chunk_end_time,
                )
            )
            prior_chunk_context = " ".join(
                current_chunk.split()[-PRIOR_CHUNK_CONTEXT_LENGTH:]
            )
            current_chunk = ""
            current_chunk_word_count = 0
            current_chunk_start_time = 0.0
            current_chunk_end_time = 0.0

        # If we have text in a segment, process and append to chunk
        if segment["text"]:
            current_chunk += segment["text"]
            current_chunk_word_count += len(segment["text"].split())
            current_chunk_end_time = segment["end"]

            if current_chunk_start_time == 0.0:
                current_chunk_start_time = segment["start"]

            if segment["text"][-1] == ".":
                previous_segment_complete_sentence = True
            else:
                previous_segment_complete_sentence = False

    # Append final chunk
    if current_chunk:
        chunks.append(
            Chunk(
                chunk=current_chunk,
                prior_chunk_context=prior_chunk_context,
                word_count=current_chunk_word_count,
                start_time=current_chunk_start_time,
                end_time=current_chunk_end_time,
            )
        )

    return chunks


def summarize_chunk(llm: OpenAI, chunk: Chunk):
    """Generate a chunk summary using an LLM"""

    completion = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You will be receiving chunks of a lecture transcript, and you should summarize them in markdown. "
                    "The goal is to have thorough notes that capture the key ideas, points, equations, or concepts of the lecture. "
                    "Don't generate headings. Focus more on capturing the core content. "
                    "Bold important terms or theories. Use italics for examples to distinguish it from the main points."
                    "Each chunk has 2 sections. The first section will start with the label PREVIOUS_CONTEXT. The text that follows should "
                    "not be summarized and is a portion of the prior chunk to give added context. "
                    "The next section starts with CURRENT_CONTEXT, and this is the part that should be explicitly summarized."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"PREVIOUS_CONTEXT: {chunk.prior_chunk_context}\n"
                    f"CURRENT_CONTEXT: {chunk.chunk}"
                ),
            },
        ],
    )

    return SummaryChunk(
        summary_chunk=completion.choices[0].message.content or "",
        transcript_chunk=chunk.chunk,
        start_time=chunk.start_time,
        end_time=chunk.end_time,
    )


def summarize_transcript(
    open_ai_client: OpenAI, transcript_filepath: str, output_filepath: str
):
    """Summarize a transcript into summary chunks"""

    chunked_transcript = chunk_transcript(transcript_filepath)
    summaries: list[SummaryChunk] = []
    for chunk in tqdm(chunked_transcript, desc="Summarizing chunks.."):
        summaries.append(summarize_chunk(open_ai_client, chunk))

    dict_summaries = [asdict(summary) for summary in summaries]
    with open(output_filepath, "w") as f:
        json.dump(dict_summaries, f, indent=2)


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

    DOWNLOADS_FOLDER = "downloads"
    TRANSCRIPTS_FOLDER = "transcripts"
    SUMMARIES_FOLDER = "summaries"

    if url is not None:
        print(f"Downloading video and audio from URL: {url}")
        video_path = download_video(url, DOWNLOADS_FOLDER)
        audio_path = download_audio(url, DOWNLOADS_FOLDER)

    if video_path is not None:
        print(video_path)
        if not video_path.endswith(".mp4"):
            raise Exception("Video file must be mp4")

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


if __name__ == "__main__":
    main()
