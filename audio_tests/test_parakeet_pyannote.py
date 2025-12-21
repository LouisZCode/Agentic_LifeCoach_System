"""
Test script for local Parakeet-MLX + Pyannote diarization.
Fully local solution - no API calls, no file size limits.

Usage:
    1. Drop an audio file in audio_sample/
    2. Run: python test_parakeet_pyannote.py
    3. Check test_results/ for output

Requirements:
    - parakeet-mlx installed
    - pyannote.audio installed
    - HUGGINGFACE_TOKEN in .env (for Pyannote)
    - FFmpeg installed (brew install ffmpeg)
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


# Pricing - local models are FREE
COST_PER_MINUTE = 0.0  # Free!


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


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in megabytes."""
    return file_path.stat().st_size / (1024 * 1024)


def transcribe_with_parakeet(audio_path: Path) -> dict:
    """
    Transcribe audio using local Parakeet-MLX model.

    Returns dict with:
        - segments: list of {text, start, end}
        - full_text: complete transcription
        - duration: audio duration
    """
    from parakeet_mlx import from_pretrained

    model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
    result = model.transcribe(str(audio_path))

    # Extract segments with timestamps
    segments = []
    for sentence in result.sentences:
        segments.append({
            "text": sentence.text.strip(),
            "start": sentence.start,
            "end": sentence.end
        })

    duration = get_audio_duration(audio_path)

    return {
        "segments": segments,
        "full_text": result.text,
        "duration": duration
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
        token=token
    )

    print("  Running diarization...")
    diarization = pipeline(audio_input)

    # Extract segments
    segments = []
    for turn, speaker in diarization.speaker_diarization:
        speaker_num = speaker.split('_')[-1] if '_' in str(speaker) else str(speaker)
        segments.append({
            "speaker": f"Speaker {speaker_num}",
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
                 output_folder: Path, had_diarization: bool,
                 timing: dict = None):
    """Save transcription results to a text file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_folder / f"parakeet_pyannote_{timestamp}_{audio_name}.txt"

    diarization_note = "Yes (Pyannote local)" if had_diarization else "No (Pyannote unavailable)"

    # Format timing info
    timing_info = ""
    if timing:
        timing_info = f"""
Processing Time:
  - Parakeet Transcription: {timing.get('parakeet', 0):.1f}s
  - Pyannote Diarization: {timing.get('pyannote', 0):.1f}s
  - Total: {timing.get('total', 0):.1f}s
  - Speed: {duration / timing.get('total', 1):.1f}x realtime
"""

    content = f"""=== TRANSCRIPTION TEST RESULTS ===
Method: Local Parakeet-MLX + Pyannote
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Audio File: {audio_name}
Audio Duration: {format_duration(duration)}

Cost: $0.00 (fully local - FREE!)
{timing_info}
Speaker Diarization: {diarization_note}

Advantages:
  - No file size limits
  - No audio length limits
  - No API costs
  - Works offline
  - Privacy (audio never leaves machine)

Disadvantages:
  - Slower than cloud APIs
  - Requires local GPU/CPU resources

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

    # Find audio file
    audio_file = find_audio_file(sample_folder)
    if not audio_file:
        print(f"Error: No audio file found in {sample_folder}")
        print("Supported formats: mp3, wav, m4a, flac, ogg, webm")
        sys.exit(1)

    print(f"Found audio file: {audio_file.name}")
    file_size = get_file_size_mb(audio_file)
    print(f"File size: {file_size:.1f}MB")
    print()

    try:
        timing = {}
        total_start = time.time()

        # Step 1: Transcribe with Parakeet
        print("Step 1: Transcribing with Parakeet-MLX (local)...")
        step1_start = time.time()
        parakeet_result = transcribe_with_parakeet(audio_file)
        timing['parakeet'] = time.time() - step1_start
        print(f"  Got {len(parakeet_result['segments'])} segments ({timing['parakeet']:.1f}s)")

        # Step 2: Diarize with Pyannote
        had_diarization = False
        speaker_segments = []
        timing['pyannote'] = 0

        if os.getenv("HUGGINGFACE_TOKEN"):
            try:
                print()
                print("Step 2: Speaker diarization with Pyannote (local)...")
                step2_start = time.time()
                speaker_segments = diarize_with_pyannote(audio_file)
                timing['pyannote'] = time.time() - step2_start
                print(f"  Found {len(set(s['speaker'] for s in speaker_segments))} speakers ({timing['pyannote']:.1f}s)")
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

        if parakeet_result['segments']:
            final_text = align_transcription_with_speakers(
                parakeet_result['segments'],
                speaker_segments
            )
        else:
            final_text = parakeet_result['full_text']

        timing['total'] = time.time() - total_start

        # Save results
        duration = parakeet_result['duration']
        output_file = save_results(
            audio_file.stem,
            final_text,
            duration,
            results_folder,
            had_diarization,
            timing
        )

        print()
        print(f"Audio duration: {format_duration(duration)}")
        print(f"Processing time: {timing['total']:.1f}s ({duration / timing['total']:.1f}x realtime)")
        print(f"Cost: $0.00 (FREE - local processing)")
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
