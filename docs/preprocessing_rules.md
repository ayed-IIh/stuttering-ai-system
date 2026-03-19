# Audio Preprocessing Rules

## The Preprocessing Pipeline

Steps are applied in order. Steps 1–3 and 5–6 run at both training and inference. Step 4 (silence trimming) is training-only — see the rationale below.

**Step 1 — Resample to 16000 Hz.**
Wav2Vec2 and HuBERT were pretrained on 16 kHz audio. Feeding them a different sample rate doesn't raise an error — it silently degrades accuracy because the model's learned feature representations assume 16 kHz frame rates.

**Step 2 — Stereo to mono by channel mean.**
`np.mean(waveform, axis=0)`. Mean over left-channel-drop because it preserves total energy and avoids bias toward one mic in dual-mic clinical recordings. If a file is already mono, this step is a no-op.

**Step 3 — Peak normalization to [-1.0, 1.0].**
`waveform / (np.abs(waveform).max() + 1e-8)`. The epsilon guards against division by zero on all-zero clips. Peak normalization is the right choice here — not RMS — because LibriSpeech (the pretraining corpus for both Wav2Vec2 and HuBERT) uses peak normalization. Switching to RMS would shift the input distribution away from what the encoder learned and would only be appropriate if training from scratch.

**Step 4 — Silence trimming (training only).**
`librosa.effects.trim(top_db=30)`, applied to leading and trailing silence. This is disabled at inference. Clinical recordings sometimes carry meaningful silence before a block begins; trimming it at inference would lose a signal that matters to the clinician reviewing the output.

**Step 5 — Fixed duration: 10.0 seconds (160,000 samples).**
All clips are brought to exactly 160,000 samples before being fed to the model.

- Long clips: center crop. `start = (n - 160_000) // 2`, then `waveform[start : start + 160_000]`. Center crop keeps the core utterance — the first and last ~2 seconds of a recording are usually silence or mic noise, so discarding them symmetrically is the right default.
- Short clips: zero-pad on the right. `np.pad(waveform, (0, pad_len))`. Right-padding matches the HuggingFace attention mask convention: padding is always appended at the end, never prepended.

**Step 6 — Cast to float32.**
`waveform.astype(np.float32)`. PyTorch's default compute dtype. No precision is lost relative to float64 for audio signals in the [-1.0, 1.0] range.

---

## Output Contract

What Adan's model receives for every sample, at both training and inference:

|    Field    |              Value            |
|-------------|-------------------------------|
|    Shape    | `(1, 160000)` — channel-first |
|    dtype    |          `torch.float32`      |
| Value range |          `[-1.0, 1.0]`        |
| Sample rate |            16000 Hz           |
|  Channels   |            1 (mono)           |

This table is written to be asserted against. An assert at the model input boundary is a reasonable sanity check.

---

## Training-Only Augmentations

Applied after all normalization steps (after Step 6), not at validation or inference. Each augmentation is drawn independently with its own probability — they are not chained in a fixed order, and multiple can apply to the same sample in a single training step.

|  Augmentation  |   p  |        Parameters          |             Call               |
|----------------|------|----------------------------|--------------------------------|
| Gaussian noise | 0.30 |       SNR 20–40 dB         |             numpy              |
|  Time stretch  | 0.20 |     rate ∈ [0.9, 1.1]      | `librosa.effects.time_stretch` |
|  Pitch shift   | 0.20 | steps ∈ [-2, +2] semitones | `librosa.effects.pitch_shift`  |
| Volume scaling | 0.30 |     gain ∈ [0.8, 1.2]      | `waveform × gain`              |

After augmentation, clips are re-cropped or re-padded to 160,000 samples if time stretch changed the length.

---

## Stability Under Dataset Growth

Adding more speakers, recording sessions, or audio conditions doesn't change any of the above. The rules describe transforms, not data properties. The only change that would force a revision here is a model architecture switch — for example, moving to a spectrogram-based model (Whisper encoder, a mel-filterbank CNN) would require revisiting the silence trimming threshold and whether 10 seconds is still the right max duration. A new Wav2Vec2 or HuBERT checkpoint with the same 16 kHz assumption changes nothing.

---

## Implementation Reference

All steps above are the responsibility of `ai/preprocessing/audio_loader.py`. The load function handles resampling (Step 1), mono conversion (Step 2), and normalization (Step 3). Silence trimming (Step 4) is gated by a `training` flag passed into the loader. Fixed-duration enforcement (Step 5) and dtype casting (Step 6) are the final operations before the tensor is returned. Augmentations live in a separate augmentation module called from the training dataset class, not from the loader itself.
