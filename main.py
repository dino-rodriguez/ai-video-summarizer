import json
import os

import click
import mlx_whisper
import yt_dlp


def download_video(url: str, output_folder: str):
    """Download video-only in MP4 format (up to 720p)"""

    ydl_opts_video = {
        "outtmpl": f"./{output_folder}/%(title)s.video.%(ext)s",  # Save video file with .video extension
        "format": "bestvideo[ext=mp4][height<=720]",  # Download best video-only (up to 720p)
        "noplaylist": True,  # Single video only
    }
    with yt_dlp.YoutubeDL(ydl_opts_video) as ydl:
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)
        return path


def download_audio(url: str, output_folder: str):
    """Download audio-only in M4A format"""

    ydl_opts_audio = {
        "outtmpl": f"./{output_folder}/%(title)s.audio.%(ext)s",  # Save audio file with .audio extension
        "format": "bestaudio[ext=m4a]",  # Download best audio-only
        "noplaylist": True,  # Single video only
    }
    with yt_dlp.YoutubeDL(ydl_opts_audio) as ydl:
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)
        return path


def transcribe_audio(audio_filepath: str):
    """Transcribe audio file to text using OpenAI Whisper"""

    result = mlx_whisper.transcribe(
        audio_filepath, path_or_hf_repo="mlx-community/whisper-turbo", verbose=False
    )

    TRANSCRIPTS_FOLDER = "transcripts"
    os.makedirs(TRANSCRIPTS_FOLDER, exist_ok=True)

    filename = audio_filepath.split("/")[-1].rstrip(".audio.m4a") + ".txt"
    file_path = os.path.join(TRANSCRIPTS_FOLDER, filename)

    with open(file_path, "w") as f:
        json.dump(result, f, indent=2)
        return file_path


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
    chunks = []

    for segment in transcript["segments"]:
        # when we are above the chunk length and just finished a sentence,
        # append and reset to new chunk
        if (
            current_chunk_word_count >= CHUNK_LENGTH
            and previous_segment_complete_sentence
        ):
            chunks.append(
                {
                    "chunk": current_chunk,
                    "prior_chunk_context": prior_chunk_context,
                    "word_count": current_chunk_word_count,
                    "start_time": current_chunk_start_time,
                    "end_time": current_chunk_end_time,
                }
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
            {
                "chunk": current_chunk,
                "prior_chunk_context": prior_chunk_context,
                "word_count": current_chunk_word_count,
                "start_time": current_chunk_start_time,
                "end_time": current_chunk_end_time,
            }
        )

    return chunks


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

    DOWNLOADS_FOLDER = "downloads"

    if url is not None:
        print(f"Downloading video and audio from URL: {url}")
        video_path = download_video(url, DOWNLOADS_FOLDER)
        audio_path = download_audio(url, DOWNLOADS_FOLDER)

    if video_path is not None:
        print(video_path)
        if not video_path.endswith(".mp4"):
            raise Exception("Video file must be mp4")

    if audio_path is not None:
        print(audio_path)
        if not audio_path.endswith(".m4a"):
            raise Exception("Audio file must be m4a")

        print(f"Transcribing audio file: {audio_path}")
        transcript_path = transcribe_audio(audio_path)
        print(f"Generated transcript: {transcript_path}")

    if transcript_path is not None:
        print(f"Chunking transcript: {transcript_path}")
        chunked_transcript = chunk_transcript(transcript_path)
        print(json.dumps(chunked_transcript, indent=2))


if __name__ == "__main__":
    main()
