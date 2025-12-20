"""
Test script for Mistral Voxtral + Pyannote diarization.
Voxtral for transcription, Pyannote for speaker identification.

Usage:
    1. Add MISTRAL_API_KEY to .env
    2. Drop an audio file in audio_sample/
    3. Run: python test_voxtral_pyannote.py
    4. Check test_results/ for output

Requirements:
    - MISTRAL_API_KEY in .env
    - HUGGINGFACE_TOKEN in .env (for Pyannote)
    - pyannote.audio installed
"""

import os
import sys
import json
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")


# Pricing (as of Dec 2025)
VOXTRAL_PRICE_PER_MINUTE = 0.001  # $0.001/min - very cheap!


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


def transcribe_with_voxtral(audio_path: Path) -> dict:
    """
    Transcribe audio using Mistral's Voxtral API.

    Returns dict with:
        - segments: list of {text, start, end}
        - full_text: complete transcription
        - duration: audio duration
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY not found in .env")

    # Mistral audio transcription endpoint
    url = "https://api.mistral.ai/v1/audio/transcriptions"

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    # Read the audio file
    with open(audio_path, "rb") as f:
        files = {
            "file": (audio_path.name, f, "audio/mpeg")
        }
        data = {
            "model": "voxtral-mini-latest",
            "timestamp_granularities": "segment",
            "response_format": "verbose_json"
        }

        print("  Sending to Voxtral API...")
        response = requests.post(url, headers=headers, files=files, data=data)

    if response.status_code != 200:
        raise RuntimeError(f"Voxtral API error: {response.status_code} - {response.text}")

    result = response.json()

    # Parse response
    segments = []
    if "segments" in result:
        for seg in result["segments"]:
            segments.append({
                "text": seg.get("text", "").strip(),
                "start": seg.get("start", 0),
                "end": seg.get("end", 0)
            })

    duration = result.get("duration", get_audio_duration(audio_path))

    return {
        "segments": segments,
        "full_text": result.get("text", ""),
        "duration": duration,
        "raw_response": result
    }


def diarize_with_pyannote(audio_path: Path) -> list[dict]:
    """
    Perform speaker diarization using Pyannote.

    Returns list of speaker segments: {speaker, start, end}
    """
    from pyannote.audio import Pipeline
    import torch
    import torchaudio

    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACE_TOKEN not found in .env")

    print("  Loading Pyannote pipeline...")

    # Load audio
    waveform, sample_rate = torchaudio.load(str(audio_path))

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    audio_input = {"waveform": waveform, "sample_rate": sample_rate}

    # Load pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token
    )

    print("  Running diarization...")
    diarization = pipeline(audio_input)

    # Extract segments
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": f"Speaker {speaker.split('_')[-1] if '_' in speaker else speaker}",
            "start": turn.start,
            "end": turn.end
        })

    return segments


def align_transcription_with_speakers(
    text_segments: list[dict],
    speaker_segments: list[dict]
) -> str:
    """
    Align transcribed text with speaker diarization.
    Uses midpoint matching.
    """
    if not speaker_segments:
        return "\n".join(seg["text"] for seg in text_segments)

    aligned = []
    current_speaker = None
    current_text = []

    for text_seg in text_segments:
        # Find speaker at midpoint
        midpoint = (text_seg["start"] + text_seg["end"]) / 2
        speaker = "UNKNOWN"

        for spk_seg in speaker_segments:
            if spk_seg["start"] <= midpoint <= spk_seg["end"]:
                speaker = spk_seg["speaker"]
                break

        # Group by speaker
        if speaker == current_speaker:
            current_text.append(text_seg["text"])
        else:
            if current_speaker and current_text:
                aligned.append(f"{current_speaker}: {' '.join(current_text)}")
            current_speaker = speaker
            current_text = [text_seg["text"]]

    # Don't forget last speaker
    if current_speaker and current_text:
        aligned.append(f"{current_speaker}: {' '.join(current_text)}")

    return "\n\n".join(aligned)


def save_results(audio_name: str, transcription: str, duration: float,
                 output_folder: Path, had_diarization: bool):
    """Save transcription results to a text file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_folder / f"voxtral_pyannote_{timestamp}_{audio_name}.txt"

    voxtral_cost = (duration / 60) * VOXTRAL_PRICE_PER_MINUTE
    pyannote_cost = 0  # Free (local)
    total_cost = voxtral_cost

    diarization_note = "Yes (Pyannote local)" if had_diarization else "No (Pyannote unavailable)"

    content = f"""=== TRANSCRIPTION TEST RESULTS ===
Method: Mistral Voxtral + Pyannote
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Audio File: {audio_name}
Duration: {format_duration(duration)}

Cost Breakdown:
  - Voxtral: ${voxtral_cost:.4f} (@ ${VOXTRAL_PRICE_PER_MINUTE}/min)
  - Pyannote: $0.00 (local/free)
  - Total: ${total_cost:.4f}

Speaker Diarization: {diarization_note}

=== TRANSCRIPTION ===

{transcription}
"""

    output_file.write_text(content, encoding='utf-8')
    return output_file


def main():
    # Setup paths
    script_dir = Path(__file__).parent
    sample_folder = script_dir / "audio_sample"
    results_folder = script_dir / "test_results"

    # Check API keys
    if not os.getenv("MISTRAL_API_KEY"):
        print("Error: MISTRAL_API_KEY not found in .env")
        print("Add your Mistral API key to the .env file")
        sys.exit(1)

    # Find audio file
    audio_file = find_audio_file(sample_folder)
    if not audio_file:
        print(f"Error: No audio file found in {sample_folder}")
        print("Supported formats: mp3, wav, m4a, flac, ogg, webm")
        sys.exit(1)

    print(f"Found audio file: {audio_file.name}")
    print()

    try:
        # Step 1: Transcribe with Voxtral
        print("Step 1: Transcribing with Voxtral...")
        voxtral_result = transcribe_with_voxtral(audio_file)
        print(f"  Got {len(voxtral_result['segments'])} segments")

        # Step 2: Diarize with Pyannote
        had_diarization = False
        speaker_segments = []

        if os.getenv("HUGGINGFACE_TOKEN"):
            try:
                print()
                print("Step 2: Speaker diarization with Pyannote...")
                speaker_segments = diarize_with_pyannote(audio_file)
                print(f"  Found {len(set(s['speaker'] for s in speaker_segments))} speakers")
                had_diarization = True
            except Exception as e:
                print(f"  Diarization failed: {e}")
                print("  Continuing without speaker labels...")
        else:
            print()
            print("Step 2: Skipping diarization (no HUGGINGFACE_TOKEN)")

        # Step 3: Align and format
        print()
        print("Step 3: Aligning transcription with speakers...")

        if voxtral_result['segments']:
            final_text = align_transcription_with_speakers(
                voxtral_result['segments'],
                speaker_segments
            )
        else:
            # Fallback to plain text
            final_text = voxtral_result['full_text']

        # Save results
        duration = voxtral_result['duration']
        output_file = save_results(
            audio_file.stem,
            final_text,
            duration,
            results_folder,
            had_diarization
        )

        print()
        print(f"Duration: {format_duration(duration)}")
        print(f"Cost estimate: ${(duration / 60) * VOXTRAL_PRICE_PER_MINUTE:.4f}")
        print()
        print(f"Results saved to: {output_file}")
        print()
        print("=== PREVIEW ===")
        print(final_text[:500] + "..." if len(final_text) > 500 else final_text)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
