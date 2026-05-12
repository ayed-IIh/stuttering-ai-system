from __future__ import annotations

import asyncio
import importlib
import io
import json
import logging
import math
import struct
import threading
import time
import wave
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torchaudio
import torchaudio.functional as TAF

from ai.preprocessing.audio_loader import (
    TARGET_DURATION_SEC,
    normalize_waveform,
    pad_or_truncate,
)
from backend.app.config import Settings
from shared.labels import CLASS_LABELS

logger = logging.getLogger(__name__)

try:
    import torch
except Exception:  # pragma: no cover - optional dependency for local dev
    torch = None  # type: ignore[assignment]

try:
    from transformers import Wav2Vec2Processor
except Exception:  # pragma: no cover - optional dependency for local dev
    Wav2Vec2Processor = None  # type: ignore[assignment]


class InvalidAudioError(Exception):
    """Raised when uploaded audio is corrupt or cannot be decoded."""


class ModelNotLoadedError(Exception):
    """Raised when model initialization fails or predict() is called before load."""


class PredictionError(Exception):
    """Raised when an inference-time failure occurs."""


class _IdentityProcessor:
    """Fallback processor used only in development when transformers is unavailable."""

    def __call__(
        self,
        waveform: Any,
        *,
        sampling_rate: int,
        return_tensors: str = "pt",
        padding: bool = True,
    ) -> dict[str, Any]:
        _ = (sampling_rate, return_tensors, padding)
        if torch is not None and not torch.is_tensor(waveform):
            return {"input_values": torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)}
        return {"input_values": waveform}


class _FallbackClassifier:
    """Deterministic fallback model for non-production environments."""

    def __init__(self, num_classes: int) -> None:
        self._num_classes = max(1, num_classes)
        self._is_eval = False
        self._device = "cpu"

    def to(self, device: str) -> _FallbackClassifier:
        self._device = device
        return self

    def eval(self) -> _FallbackClassifier:
        self._is_eval = True
        return self

    def __call__(self, **_: Any) -> Any:
        logits = [float(i) for i in range(self._num_classes)]
        return {"logits": logits}


def _model_block(cfg: dict[str, Any]) -> dict[str, Any]:
    m = cfg.get("model")
    if isinstance(m, dict) and m:
        return m
    return cfg


def _data_block(cfg: dict[str, Any]) -> dict[str, Any]:
    d = cfg.get("data")
    return d if isinstance(d, dict) else {}


def _training_block(cfg: dict[str, Any]) -> dict[str, Any]:
    t = cfg.get("training")
    return t if isinstance(t, dict) else {}


def _state_dict_from_inference_payload(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    if "model_state_dict" in raw:
        sd = raw["model_state_dict"]
        return sd if isinstance(sd, dict) else None
    if "state_dict" in raw:
        sd = raw["state_dict"]
        return sd if isinstance(sd, dict) else None
    return None


class ModelService:
    """Load ADN-04 inference artifacts and run thread-safe Wav2Vec2 inference."""

    def __init__(self, config: Settings) -> None:
        """Initialize from app settings (pydantic ``Settings`` — the service ``config``)."""
        self._settings = config
        self._model: Any = None
        self._processor: Any = None
        self._loaded = False
        self._predict_lock = threading.Lock()
        self._model_version = config.SERVICE_VERSION
        self._model_config: dict[str, Any] = {}
        self._class_names: list[str] = list(CLASS_LABELS)
        self._target_sample_rate = 16_000
        self._max_duration_sec = float(TARGET_DURATION_SEC)
        self._load_sync()

    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_version(self) -> str:
        return self._model_version if self._loaded else "unknown"

    async def load(self) -> None:
        """Async hook for lifespan; sync load already ran in ``__init__``."""
        if self._loaded:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self) -> None:
        if self._loaded:
            return
        started_at = time.perf_counter()
        # trigger S3 download before resolving paths
        if self._settings.MODEL_SOURCE == "s3":
            from backend.app.config import download_model_if_needed
            download_model_if_needed(self._settings)

        artifact_dir, model_path, config_path = self._resolve_artifact_paths(str(self._settings.resolved_model_path))
        if not config_path.exists() or not model_path.exists():
            if self._settings.PRODUCTION_MODE:
                raise ModelNotLoadedError(
                    f"Missing artifact files under {artifact_dir} (need model_inference.pt and config.json)"
                )
            self._use_fallback_model("missing_artifacts")
            self._log_load_time(started_at, source="fallback")
            return

        self._model_config = self._read_json_config(config_path)
        self._class_names = self._extract_class_names(self._model_config)
        data_cfg = _data_block(self._model_config)
        self._target_sample_rate = int(
            data_cfg.get("target_sr")
            or self._model_config.get("target_sample_rate")
            or self._model_config.get("sample_rate")
            or 16_000
        )
        self._max_duration_sec = float(
            data_cfg.get("max_duration_sec")
            or self._model_config.get("max_duration_sec")
            or TARGET_DURATION_SEC
        )
        self._model_version = str(
            self._model_config.get("model_version", self._settings.SERVICE_VERSION)
        )
        self._processor = self._load_processor(self._model_config)
        self._model = self._load_model(model_path, self._model_config)
        self._loaded = True
        self._log_load_time(started_at, source="artifact")

    def _resolve_artifact_paths(self, model_path_value: str) -> tuple[Path, Path, Path]:
        model_path = Path(model_path_value or "").expanduser()
        if model_path.is_file():
            artifact_dir = model_path.parent
        else:
            artifact_dir = model_path if model_path_value else Path.cwd()
        return artifact_dir, artifact_dir / "model_inference.pt", artifact_dir / "config.json"

    def _read_json_config(self, config_path: Path) -> dict[str, Any]:
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ModelNotLoadedError(f"Failed to parse config.json: {exc}") from exc

    def _extract_class_names(self, cfg: dict[str, Any]) -> list[str]:
        label2id = cfg.get("label2id")
        if isinstance(label2id, dict) and label2id:
            pairs = sorted(label2id.items(), key=lambda x: int(x[1]))
            return [str(k) for k, _ in pairs]
        for key in ("class_names", "labels", "classes"):
            value = cfg.get(key)
            if isinstance(value, list) and value:
                return [str(v) for v in value]
        mb = _model_block(cfg)
        for key in ("class_names", "labels", "classes"):
            value = mb.get(key)
            if isinstance(value, list) and value:
                return [str(v) for v in value]
        return list(self._class_names)

    def _load_processor(self, cfg: dict[str, Any]) -> Any:
        mb = _model_block(cfg)
        model_name = str(mb.get("model_name", cfg.get("model_name", ""))).strip()
        if not model_name:
            if self._settings.PRODUCTION_MODE:
                raise ModelNotLoadedError("config.json is missing required field 'model_name'")
            return _IdentityProcessor()
        if Wav2Vec2Processor is None:
            if self._settings.PRODUCTION_MODE:
                raise ModelNotLoadedError("transformers is not installed; cannot load Wav2Vec2Processor")
            return _IdentityProcessor()
        try:
            return Wav2Vec2Processor.from_pretrained(model_name)
        except Exception as exc:
            if self._settings.PRODUCTION_MODE:
                raise ModelNotLoadedError(f"Failed to load Wav2Vec2Processor: {exc}") from exc
            return _IdentityProcessor()

    def _load_model(self, model_path: Path, cfg: dict[str, Any]) -> Any:
        if torch is None:
            if self._settings.PRODUCTION_MODE:
                raise ModelNotLoadedError("torch is not installed; cannot load model_inference.pt")
            return _FallbackClassifier(num_classes=len(self._class_names)).eval()

        model: Any = self._try_load_stuttering_classifier(model_path, cfg)
        if model is None:
            classifier_class = cfg.get("classifier_class")
            if isinstance(classifier_class, str) and classifier_class.strip():
                model = self._rebuild_classifier(classifier_class, cfg, model_path)
        if model is None:
            model = self._load_torchscript(model_path)
        if model is None:
            if self._settings.PRODUCTION_MODE:
                raise ModelNotLoadedError(f"Unable to reconstruct model from {model_path}")
            model = _FallbackClassifier(num_classes=len(self._class_names))

        device = self._settings.DEVICE
        if hasattr(model, "to"):
            model = model.to(device)
        if hasattr(model, "eval"):
            model.eval()
        return model

    def _try_load_stuttering_classifier(self, model_path: Path, cfg: dict[str, Any]) -> Any:
        if torch is None:
            return None
        try:
            from ai.models.stuttering_classifier import ModelConfig, StutteringClassifier
        except Exception:
            return None

        mb = _model_block(cfg)
        if not mb.get("model_name"):
            return None

        tr = _training_block(cfg)
        try:
            mc = ModelConfig(
                model_name=str(mb["model_name"]),
                num_classes=int(mb.get("num_classes", len(self._class_names))),
                dropout_rate=float(mb.get("dropout_rate", 0.1)),
                freeze_encoder=bool(mb.get("freeze_encoder", True)),
                learning_rate=float(tr.get("learning_rate", mb.get("learning_rate", 1e-4))),
            )
            model = StutteringClassifier(mc)
        except Exception:
            return None

        try:
            raw = torch.load(model_path, map_location="cpu", weights_only=False)
        except Exception:
            return None
        if not isinstance(raw, dict):
            return None

        sd = raw.get("model_state_dict") or raw.get("state_dict")
        if sd is None:
            if any(k in raw for k in ("optimizer_state_dict", "epoch", "best_val_loss")):
                return None
            sd = raw
        if not isinstance(sd, dict):
            return None

        try:
            model.load_state_dict(sd, strict=True)
        except Exception:
            try:
                model.load_state_dict(sd, strict=False)
            except Exception:
                return None

        return model

    def _rebuild_classifier(self, dotted_path: str, cfg: dict[str, Any], model_path: Path) -> Any:
        if torch is None:
            return None
        try:
            module_name, class_name = dotted_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            classifier_cls = getattr(module, class_name)
            init_args = cfg.get("model_init_args", {})
            model = classifier_cls(**init_args)
            checkpoint = torch.load(model_path, map_location=self._settings.DEVICE, weights_only=False)
            if isinstance(checkpoint, dict):
                sd = _state_dict_from_inference_payload(checkpoint)
                if sd is None:
                    sd = checkpoint.get("state_dict", checkpoint)
            else:
                sd = checkpoint
            if hasattr(model, "load_state_dict") and isinstance(sd, dict):
                model.load_state_dict(sd)
            return model
        except Exception:
            return None

    def _load_torchscript(self, model_path: Path) -> Any:
        if torch is None:
            return None
        try:
            return torch.jit.load(str(model_path), map_location=self._settings.DEVICE)
        except Exception:
            return None

    def _use_fallback_model(self, reason: str) -> None:
        self._model_config = {"fallback_reason": reason}
        self._processor = _IdentityProcessor()
        self._model = _FallbackClassifier(num_classes=len(self._class_names)).eval()
        self._loaded = True
        self._model_version = self._settings.SERVICE_VERSION

    def _log_load_time(self, started_at: float, *, source: str) -> None:
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        logger.info(
            "model_loaded",
            extra={
                "source": source,
                "device": self._settings.DEVICE,
                "model_version": self._model_version,
                "load_time_ms": elapsed_ms,
            },
        )

    def predict(self, audio_bytes: bytes) -> dict[str, Any]:
        """Run multi-label inference on a WAV payload.

        Returns a dict with:
            predicted_classes : list of ``{class, confidence}`` for every class
                                whose sigmoid probability is ``>= threshold``.
                                Empty list when no class crosses the threshold.
            all_scores        : full per-class sigmoid probabilities (do NOT
                                sum to 1.0 — each is an independent binary head).
            threshold         : the threshold actually applied (from settings).
            processing_time_ms: measured wall-clock for this call.
            model_version     : the loaded model artifact's version string.
        """
        if not self._loaded:
            raise ModelNotLoadedError("ModelService is not loaded")
        started_at = time.perf_counter()
        with self._predict_lock:
            try:
                waveform = self._decode_and_preprocess_wav(audio_bytes)
                inputs = self._prepare_inputs(waveform)
                outputs = self._forward(inputs)
                probs = self._sigmoid(outputs)
                threshold = float(self._settings.MULTI_LABEL_THRESHOLD)
                # The wire contract is keyed by canonical CLASS_LABELS; do not
                # silently slice or remap — drift here would attach probs to
                # the wrong class names. Fail fast if the loaded artifact's
                # class order disagrees with the shared taxonomy.
                if (
                    list(self._class_names) != list(CLASS_LABELS)
                    or len(probs) != len(CLASS_LABELS)
                ):
                    raise PredictionError(
                        "Model artifact class order does not match "
                        "shared.labels.CLASS_LABELS — refusing to assign "
                        "probabilities to potentially wrong class names."
                    )
                all_scores = {
                    name: float(probs[i]) for i, name in enumerate(CLASS_LABELS)
                }
                predicted_classes = [
                    {"class": name, "confidence": float(probs[i])}
                    for i, name in enumerate(CLASS_LABELS)
                    if probs[i] >= threshold
                ]
                # Sort detected classes by confidence descending for stable client UX.
                predicted_classes.sort(key=lambda d: d["confidence"], reverse=True)
                elapsed_ms = int((time.perf_counter() - started_at) * 1000)
                return {
                    "predicted_classes": predicted_classes,
                    "all_scores": all_scores,
                    "threshold": threshold,
                    "processing_time_ms": elapsed_ms,
                    "model_version": self._model_version,
                }
            except (InvalidAudioError, ModelNotLoadedError):
                raise
            except Exception as exc:
                raise PredictionError(f"Inference failed: {exc}") from exc

    def _decode_and_preprocess_wav(self, audio_bytes: bytes) -> Any:
        """Decode WAV bytes and apply the same core steps as ``audio_loader.load_audio`` (inference: no trim)."""
        if not audio_bytes:
            raise InvalidAudioError("Audio payload is empty")

        if torch is None:
            return self._decode_wav_bytes_wave_only(audio_bytes)

        try:
            waveform, native_sr = torchaudio.load(io.BytesIO(audio_bytes), format="wav")
        except Exception as exc:
            raise InvalidAudioError(f"Invalid or undecodable WAV: {exc}") from exc

        waveform = waveform.to(torch.float32)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        target_sr = self._target_sample_rate
        if int(native_sr) != target_sr:
            waveform = TAF.resample(waveform, orig_freq=int(native_sr), new_freq=target_sr)

        waveform = normalize_waveform(waveform, method="peak")
        waveform = pad_or_truncate(
            waveform, sr=target_sr, max_duration_sec=self._max_duration_sec
        )
        return waveform.to(torch.float32)

    def _decode_wav_bytes_wave_only(self, audio_bytes: bytes) -> Any:
        """Fallback when torch is unavailable (tests / minimal env)."""
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                num_frames = wav_file.getnframes()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                raw = wav_file.readframes(num_frames)
        except wave.Error as exc:
            raise InvalidAudioError(f"Invalid WAV audio: {exc}") from exc

        samples = self._pcm_to_floats(raw, sample_width)
        if channels > 1:
            mono: list[float] = []
            for i in range(0, len(samples), channels):
                frame = samples[i:i + channels]
                if not frame:
                    continue
                mono.append(sum(frame) / float(len(frame)))
            samples = mono
        return samples, sample_rate

    def _pcm_to_floats(self, raw: bytes, sample_width: int) -> list[float]:
        if sample_width == 1:
            return [((b - 128) / 128.0) for b in raw]
        if sample_width == 2:
            count = len(raw) // 2
            values = struct.unpack("<" + ("h" * count), raw)
            return [v / 32768.0 for v in values]
        if sample_width == 4:
            count = len(raw) // 4
            values = struct.unpack("<" + ("i" * count), raw)
            return [v / 2147483648.0 for v in values]
        raise InvalidAudioError(f"Unsupported WAV sample width: {sample_width}")

    def _prepare_inputs(self, waveform: Any) -> dict[str, Any]:
        processor = self._processor
        if processor is None:
            raise ModelNotLoadedError("Wav2Vec2Processor is not initialized")

        if torch is not None and torch.is_tensor(waveform):
            values = waveform.squeeze(0).detach().cpu().tolist()
            inputs = processor(
                values,
                sampling_rate=self._target_sample_rate,
                return_tensors="pt",
                padding=True,
            )
        elif isinstance(waveform, tuple):
            samples, sr = waveform
            if torch is not None:
                tensor = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)
                tensor = normalize_waveform(tensor, method="peak")
                tensor = pad_or_truncate(
                    tensor, sr=int(sr), max_duration_sec=self._max_duration_sec
                )
                values = tensor.squeeze(0).tolist()
                sr = int(sr)
            else:
                values = samples
                sr = int(sr)
            inputs = processor(
                values,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
            )
        else:
            values = waveform
            inputs = processor(
                values,
                sampling_rate=self._target_sample_rate,
                return_tensors="pt",
                padding=True,
            )

        if not isinstance(inputs, dict):
            raise PredictionError("Processor returned invalid payload")
        if torch is not None:
            for key, value in list(inputs.items()):
                if torch.is_tensor(value):
                    inputs[key] = value.to(self._settings.DEVICE)
        return inputs

    def _forward(self, inputs: dict[str, Any]) -> Any:
        if self._model is None:
            raise ModelNotLoadedError("Model is not initialized")
        no_grad = torch.no_grad if torch is not None else nullcontext
        with no_grad():
            output = self._model(**inputs)
        logits = getattr(output, "logits", None)
        if logits is None and isinstance(output, dict):
            logits = output.get("logits")
        if logits is None:
            logits = output
        return logits

    def _sigmoid(self, logits: Any) -> list[float]:
        """Per-class sigmoid for multi-label decoding.

        Unlike softmax, the resulting probabilities are *independent* per class
        and do NOT sum to 1.0. Accepts torch tensors, dicts with ``"logits"``,
        nested lists, or flat lists/tuples — same broad input shapes the old
        softmax helper supported.
        """
        if torch is not None and torch.is_tensor(logits):
            tensor = logits
            if tensor.dim() > 1:
                tensor = tensor[0]
            probs = torch.sigmoid(tensor).detach().cpu().tolist()
            return [float(p) for p in probs]
        if isinstance(logits, dict) and "logits" in logits:
            logits = logits["logits"]
        if isinstance(logits, (list, tuple)) and logits and isinstance(logits[0], (list, tuple)):
            logits = logits[0]
        arr = [float(x) for x in logits]
        if not arr:
            raise PredictionError("Model returned empty logits")
        # 1 / (1 + e^-x); clamp x to avoid overflow in math.exp.
        out: list[float] = []
        for v in arr:
            if v >= 0:
                ev = math.exp(-v)
                out.append(1.0 / (1.0 + ev))
            else:
                ev = math.exp(v)
                out.append(ev / (1.0 + ev))
        return out

    def shutdown(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False
