# Audio Transcription Tests

Compare different transcription APIs for accuracy, speed, and cost.

## Setup

1. Add API keys to `.env` (in project root):
   ```
   OPENAI_API_KEY=sk-...
   MISTRAL_API_KEY=...         # Add this for Voxtral
   HUGGINGFACE_TOKEN=hf_...    # For Pyannote diarization
   ```

2. Drop an audio file in `audio_sample/`

## Run Tests

```bash
# Option A: OpenAI gpt-4o-transcribe
python audio_tests/test_openai_diarize.py

# Option B: Voxtral + Pyannote
python audio_tests/test_voxtral_pyannote.py
```

## Results

Results are saved to `test_results/` with:
- Method used
- Audio filename
- Duration
- Cost estimate
- Full transcription with speaker labels

## Pricing Reference

| API | Price | 60 min cost |
|-----|-------|-------------|
| OpenAI gpt-4o-transcribe | $0.006/min | $0.36 |
| Mistral Voxtral | $0.001/min | $0.06 |
| Pyannote | Free (local) | $0.00 |
