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
import time
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
    Transcribe audio using OpenAI's gpt-4o-transcribe-diarize model.
    Returns transcription with speaker labels.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Get audio duration first
    duration = get_audio_duration(audio_path)

    with open(audio_path, "rb") as audio_file:
        try:
            # Use gpt-4o-transcribe-diarize with correct parameters
            response = client.audio.transcriptions.create(
                model="gpt-4o-transcribe-diarize",
                file=audio_file,
                response_format="diarized_json",  # Required for speaker labels
                chunking_strategy="auto"  # Required for audio >30 seconds
            )

            # Parse diarized_json response
            # Response has .text and .segments with speaker info
            formatted_lines = []

            if hasattr(response, 'segments') and response.segments:
                current_speaker = None
                current_text = []

                for segment in response.segments:
                    # Segments are objects, not dicts - use getattr
                    speaker = getattr(segment, 'speaker', 'UNKNOWN')
                    text = getattr(segment, 'text', '').strip()

                    if speaker == current_speaker:
                        current_text.append(text)
                    else:
                        if current_speaker is not None and current_text:
                            formatted_lines.append(f"{current_speaker}: {' '.join(current_text)}")
                        current_speaker = speaker
                        current_text = [text]

                # Don't forget last speaker
                if current_speaker is not None and current_text:
                    formatted_lines.append(f"{current_speaker}: {' '.join(current_text)}")

                formatted_text = "\n\n".join(formatted_lines)
            else:
                # Fallback to plain text if no segments
                formatted_text = response.text if hasattr(response, 'text') else str(response)

            return {
                "text": formatted_text,
                "raw_response": response,
                "duration": duration,
                "had_diarization": bool(hasattr(response, 'segments') and response.segments)
            }

        except Exception as e:
            # Fallback to basic transcription
            print(f"Note: Diarization failed, using whisper-1. Error: {e}")
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json"
            )
            return {
                "text": response.text,
                "raw_response": response,
                "duration": duration or getattr(response, 'duration', 0),
                "had_diarization": False
            }


def save_results(audio_name: str, result: dict, output_folder: Path,
                 timing: dict = None):
    """Save transcription results to a text file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_folder / f"openai_{timestamp}_{audio_name}.txt"

    duration = result.get("duration", 0)
    cost_estimate = (duration / 60) * PRICE_PER_MINUTE
    had_diarization = result.get("had_diarization", False)

    # Format timing info
    timing_info = ""
    if timing:
        timing_info = f"""
Processing Time:
  - OpenAI API: {timing.get('openai', 0):.1f}s
  - Total: {timing.get('total', 0):.1f}s
  - Speed: {duration / timing.get('total', 1):.1f}x realtime
"""

    diarization_note = "Yes (built-in)" if had_diarization else "No (fallback to Whisper)"

    content = f"""=== TRANSCRIPTION TEST RESULTS ===
Method: OpenAI gpt-4o-transcribe-diarize
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Audio File: {audio_name}
Audio Duration: {format_duration(duration)}
Cost Estimate: ${cost_estimate:.4f} (@ ${PRICE_PER_MINUTE}/min)
{timing_info}
Speaker Diarization: {diarization_note}

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
        timing = {}
        total_start = time.time()

        # Transcribe
        print("Sending to OpenAI API...")
        api_start = time.time()
        result = transcribe_with_diarization(audio_file)
        timing['openai'] = time.time() - api_start
        timing['total'] = time.time() - total_start

        # Save results
        output_file = save_results(audio_file.stem, result, results_folder, timing)

        duration = result['duration']
        print()
        print(f"Audio duration: {format_duration(duration)}")
        print(f"Processing time: {timing['total']:.1f}s ({duration / timing['total']:.1f}x realtime)")
        print(f"Cost estimate: ${(duration / 60) * PRICE_PER_MINUTE:.4f}")
        print()
        print(f"Results saved to: {output_file}")
        print()
        print("=== PREVIEW ===")
        print(result['text'][:500] + "..." if len(result['text']) > 500 else result['text'])

    except Exception as e:
        print(f"Error during transcription: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
