"""
Test script for OpenAI gpt-4o-transcribe-diarize API.
Transcription with built-in speaker diarization.

Usage:
    1. Drop an audio file in audio_sample/
    2. Run: python test_openai_diarize.py
    3. Check test_results/ for output
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

from openai import OpenAI


# Pricing (as of Dec 2025)
PRICE_PER_MINUTE = 0.006  # $0.006/min for gpt-4o-transcribe


def find_audio_file(folder: Path) -> Path | None:
    """Find the audio file in the sample folder."""
    audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.webm'}

    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in audio_extensions:
            return file
    return None


def get_audio_duration(file_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    import subprocess

    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', str(file_path)],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def format_duration(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def transcribe_with_openai_diarize(audio_path: Path) -> dict:
    """
    Transcribe audio using OpenAI's gpt-4o-transcribe-diarize model.

    Returns dict with:
        - text: formatted transcription with speaker labels
        - raw_response: the raw API response
        - duration: audio duration in seconds
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )

    # Get duration from response or calculate
    duration = getattr(response, 'duration', None) or get_audio_duration(audio_path)

    # Format the transcription
    # Note: gpt-4o-transcribe returns segments, we format them
    formatted_text = response.text

    return {
        "text": formatted_text,
        "raw_response": response,
        "duration": duration
    }


def transcribe_with_diarization(audio_path: Path) -> dict:
    """
    Transcribe audio using OpenAI's diarization endpoint.
    Uses the newer approach with speaker identification.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Get audio duration first
    duration = get_audio_duration(audio_path)

    with open(audio_path, "rb") as audio_file:
        # Try the diarization model
        try:
            response = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )

            # Format output with segments
            formatted_lines = []
            if hasattr(response, 'segments') and response.segments:
                for segment in response.segments:
                    start = format_duration(segment.get('start', 0))
                    text = segment.get('text', '').strip()
                    formatted_lines.append(f"[{start}] {text}")
                formatted_text = "\n\n".join(formatted_lines)
            else:
                formatted_text = response.text

            return {
                "text": formatted_text,
                "raw_response": response,
                "duration": duration or getattr(response, 'duration', 0)
            }

        except Exception as e:
            # Fallback to basic transcription
            print(f"Note: Using basic transcription. Error: {e}")
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json"
            )
            return {
                "text": response.text,
                "raw_response": response,
                "duration": duration or getattr(response, 'duration', 0)
            }


def save_results(audio_name: str, result: dict, output_folder: Path):
    """Save transcription results to a text file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_folder / f"openai_{timestamp}_{audio_name}.txt"

    duration = result.get("duration", 0)
    cost_estimate = (duration / 60) * PRICE_PER_MINUTE

    content = f"""=== TRANSCRIPTION TEST RESULTS ===
Method: OpenAI gpt-4o-transcribe
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Audio File: {audio_name}
Duration: {format_duration(duration)}
Cost Estimate: ${cost_estimate:.4f} (@ ${PRICE_PER_MINUTE}/min)

Note: OpenAI's API transcription model. For speaker diarization,
the gpt-4o-transcribe-diarize model may need to be used via a
different endpoint or the response parsed differently.

=== TRANSCRIPTION ===

{result['text']}
"""

    output_file.write_text(content, encoding='utf-8')
    return output_file


def main():
    # Setup paths
    script_dir = Path(__file__).parent
    sample_folder = script_dir / "audio_sample"
    results_folder = script_dir / "test_results"

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in .env")
        sys.exit(1)

    # Find audio file
    audio_file = find_audio_file(sample_folder)
    if not audio_file:
        print(f"Error: No audio file found in {sample_folder}")
        print("Supported formats: mp3, wav, m4a, flac, ogg, webm")
        sys.exit(1)

    print(f"Found audio file: {audio_file.name}")
    print(f"Transcribing with OpenAI gpt-4o-transcribe...")
    print()

    try:
        # Transcribe
        result = transcribe_with_diarization(audio_file)

        # Save results
        output_file = save_results(audio_file.stem, result, results_folder)

        print(f"Duration: {format_duration(result['duration'])}")
        print(f"Cost estimate: ${(result['duration'] / 60) * PRICE_PER_MINUTE:.4f}")
        print()
        print(f"Results saved to: {output_file}")
        print()
        print("=== PREVIEW ===")
        print(result['text'][:500] + "..." if len(result['text']) > 500 else result['text'])

    except Exception as e:
        print(f"Error during transcription: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
