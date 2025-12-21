"""
Transcription functions using Deepgram Nova-3 API.
Fast cloud-based transcription with built-in speaker diarization.

Requires:
    - DEEPGRAM_API_KEY in .env
    - deepgram-sdk installed (uv add deepgram-sdk)
"""

import os
import subprocess
from pathlib import Path

from deepgram import DeepgramClient


def is_deepgram_available() -> bool:
    """
    Check if Deepgram API is available (API key configured).

    Returns:
        bool: True if DEEPGRAM_API_KEY is set in environment.
    """
    return bool(os.getenv("DEEPGRAM_API_KEY"))


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in megabytes."""
    return file_path.stat().st_size / (1024 * 1024)


def convert_wav_to_mp3(wav_path: Path) -> Path:
    """
    Convert WAV to MP3 for faster uploads.

    Large WAV files (100MB+) can timeout during upload.
    MP3 at 128kbps is ~10x smaller and sufficient for speech.

    Args:
        wav_path: Path to the WAV file.

    Returns:
        Path to the MP3 file (created in same directory).

    Raises:
        RuntimeError: If FFmpeg conversion fails.
    """
    mp3_path = wav_path.with_suffix('.mp3')

    # Skip if MP3 already exists and is newer than WAV
    if mp3_path.exists() and mp3_path.stat().st_mtime > wav_path.stat().st_mtime:
        return mp3_path

    result = subprocess.run(
        ['ffmpeg', '-y', '-i', str(wav_path), '-codec:a', 'libmp3lame',
         '-b:a', '128k', str(mp3_path)],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")

    return mp3_path


def format_diarized_transcript(response) -> str:
    """
    Format Deepgram response into speaker-labeled transcript.

    Uses utterances if available (grouped by speaker), otherwise
    falls back to word-level speaker assignment.

    Args:
        response: Deepgram API response object.

    Returns:
        Formatted transcript with speaker labels.
    """
    results = response.results if hasattr(response, 'results') else response

    # Try utterances first (cleaner output)
    if hasattr(results, 'utterances') and results.utterances:
        lines = []
        for utterance in results.utterances:
            speaker_num = int(utterance.speaker) if utterance.speaker is not None else 0
            speaker = f"Speaker {speaker_num:02d}"
            text = utterance.transcript.strip() if utterance.transcript else ""
            if text:
                lines.append(f"{speaker}: {text}")
        return "\n\n".join(lines)

    # Fallback: use channels/alternatives with word-level diarization
    if hasattr(results, 'channels') and results.channels:
        channel = results.channels[0]
        if hasattr(channel, 'alternatives') and channel.alternatives:
            alt = channel.alternatives[0]

            if hasattr(alt, 'words') and alt.words:
                lines = []
                current_speaker = None
                current_text = []

                for word in alt.words:
                    speaker = getattr(word, 'speaker', None)

                    if speaker != current_speaker:
                        if current_speaker is not None and current_text:
                            speaker_num = int(current_speaker)
                            lines.append(f"Speaker {speaker_num:02d}: {' '.join(current_text)}")
                        current_speaker = speaker
                        current_text = [word.word]
                    else:
                        current_text.append(word.word)

                if current_speaker is not None and current_text:
                    speaker_num = int(current_speaker)
                    lines.append(f"Speaker {speaker_num:02d}: {' '.join(current_text)}")

                return "\n\n".join(lines)

            return alt.transcript if alt.transcript else "No transcription available"

    return "No transcription available"


def transcribe_with_deepgram(
    audio_path: str,
    progress_callback=None
) -> tuple[str, bool]:
    """
    Transcribe audio using Deepgram Nova-3 with speaker diarization.

    Fast cloud-based transcription (~58x realtime speed).
    Cost: $0.0043/min (~$0.26/hour).

    Args:
        audio_path: Path to the audio file.
        progress_callback: Optional callback(progress, stage) for UI updates.
            Note: Deepgram is very fast, so progress may jump quickly.

    Returns:
        tuple: (transcription_text, diarization_used)
            - transcription_text: Formatted transcript with speaker labels
            - diarization_used: Always True (built-in to Deepgram)

    Raises:
        RuntimeError: If API key not configured or API call fails.
    """
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY not found in .env")

    audio_path = Path(audio_path)

    # Report progress: starting
    if progress_callback:
        progress_callback(0.1, "preparing")

    # Convert WAV to MP3 for faster upload (100MB WAV -> ~10MB MP3)
    upload_path = audio_path
    if audio_path.suffix.lower() == '.wav':
        if progress_callback:
            progress_callback(0.2, "converting")
        upload_path = convert_wav_to_mp3(audio_path)

    # Create Deepgram client with extended timeout for large files
    client = DeepgramClient(api_key=api_key, timeout=300.0)

    # Read the audio file
    with open(upload_path, "rb") as audio_file:
        buffer_data = audio_file.read()

    # Report progress: uploading
    if progress_callback:
        progress_callback(0.4, "uploading")

    # Transcribe using Deepgram Nova-3
    response = client.listen.v1.media.transcribe_file(
        request=buffer_data,
        model="nova-3",
        smart_format=True,      # Smart punctuation and formatting
        diarize=True,           # Speaker diarization (built-in)
        punctuate=True,         # Add punctuation
        utterances=True,        # Group into utterances
        language="en",          # Language
    )

    # Report progress: formatting
    if progress_callback:
        progress_callback(0.9, "formatting")

    # Format transcription with speaker labels
    formatted_text = format_diarized_transcript(response)

    # Report progress: done
    if progress_callback:
        progress_callback(1.0, "complete")

    # Diarization is always used with Deepgram
    return formatted_text, True
