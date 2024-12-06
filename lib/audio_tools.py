import json
from dataclasses import asdict, dataclass

import mlx_whisper
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


def summarize_transcript(
    open_ai_client: OpenAI, transcript_filepath: str, output_filepath: str
):
    """Summarize a transcript into summary chunks"""

    chunked_transcript = _chunk_transcript(transcript_filepath)
    summaries: list[SummaryChunk] = []
    for chunk in tqdm(chunked_transcript, desc="Summarizing chunks.."):
        summaries.append(_summarize_chunk(open_ai_client, chunk))

    dict_summaries = [asdict(summary) for summary in summaries]
    with open(output_filepath, "w") as f:
        json.dump(dict_summaries, f, indent=2)


def transcribe_audio(audio_filepath: str, output_path: str):
    """Transcribe audio file to text using OpenAI Whisper"""

    result = mlx_whisper.transcribe(
        audio_filepath, path_or_hf_repo="mlx-community/whisper-turbo", verbose=False
    )

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)


def _chunk_transcript(transcript_filepath: str):
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


def _summarize_chunk(llm: OpenAI, chunk: Chunk):
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
