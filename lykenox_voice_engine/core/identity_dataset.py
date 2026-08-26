"""Identity voice dataset capture and metadata management."""

from __future__ import annotations

import json
import math
import shutil
import uuid
import wave
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

MODES = ("speech", "singing")
TARGET_SAMPLE_RATE = 48_000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2

DEFAULT_SPEECH_PROMPTS = (
    "Hola, esta es mi voz natural para el modelo LYKENOX.",
    "Hoy voy a leer este texto con claridad y sin forzar la voz.",
    "El objetivo es que mi identidad vocal se conserve en cada frase.",
    "Quiero que este sistema pueda leer cualquier texto en español.",
    "Esta grabación debe sonar tranquila, limpia y parecida a mí.",
)

DEFAULT_SINGING_PROMPTS = (
    "baila conmigo",
    "quédate cerca de mí",
    "hoy canta mi corazón",
    "sueño con verte otra vez",
    "mi voz se queda aquí",
)


@dataclass(frozen=True)
class IdentityPrompt:
    """One recording prompt for speech or singing identity capture."""

    id: str
    mode: str
    text: str
    language: str = "es"
    melody_hint: str | None = None


@dataclass(frozen=True)
class IdentityTakeMetadata:
    """Quality and identity metadata for one accepted or candidate take."""

    id: str
    profile: str
    mode: str
    prompt_id: str
    text: str
    language: str
    wav_path: str
    duration_sec: float
    sample_rate: int
    channels: int
    sample_width_bits: int
    rms: int
    peak: int
    clipping: bool
    measured_f0_hz: float
    measured_midi: float
    voiced_ratio: float
    status: str
    reason: str | None
    created_at: str


class IdentityDatasetService:
    """Manage the speech/singing dataset for the personal LYKENOX model."""

    def __init__(self, root: Path, profile: str = "lykenox") -> None:
        self.root = root
        self.profile = profile
        self.base_dir = root / "datasets" / profile / "identity_voice"
        self.metadata_dir = self.base_dir / "metadata"
        self.prompts_path = self.metadata_dir / "prompts.json"
        self.catalog_path = self.metadata_dir / "takes.jsonl"

    def ensure_structure(self) -> dict[str, str]:
        """Create dataset folders and starter prompts if missing."""

        for mode in MODES:
            for state in ("raw", "accepted", "rejected"):
                (self.base_dir / mode / state).mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        if not self.prompts_path.exists():
            prompts = [
                *[
                    asdict(IdentityPrompt(f"speech-{index:03d}", "speech", text))
                    for index, text in enumerate(DEFAULT_SPEECH_PROMPTS, start=1)
                ],
                *[
                    asdict(
                        IdentityPrompt(
                            f"singing-{index:03d}",
                            "singing",
                            text,
                            melody_hint="canta con una melodia comoda y natural",
                        )
                    )
                    for index, text in enumerate(DEFAULT_SINGING_PROMPTS, start=1)
                ],
            ]
            self.prompts_path.write_text(json.dumps(prompts, indent=2, ensure_ascii=False), encoding="utf-8")
        if not self.catalog_path.exists():
            self.catalog_path.write_text("", encoding="utf-8")
        return {
            "base_dir": str(self.base_dir),
            "speech_raw": str(self.base_dir / "speech" / "raw"),
            "singing_raw": str(self.base_dir / "singing" / "raw"),
            "metadata": str(self.metadata_dir),
            "prompts": str(self.prompts_path),
            "catalog": str(self.catalog_path),
        }

    def prompts(self, mode: str) -> list[IdentityPrompt]:
        """Return prompts for one capture mode."""

        self.ensure_structure()
        return [
            IdentityPrompt(**item)
            for item in json.loads(self.prompts_path.read_text(encoding="utf-8"))
            if item["mode"] == mode
        ]

    def next_prompt(self, mode: str) -> IdentityPrompt:
        """Return the first prompt with no accepted take, or the first prompt."""

        prompts = self.prompts(mode)
        accepted = {take.prompt_id for take in self.takes(mode, status="accepted")}
        for prompt in prompts:
            if prompt.id not in accepted:
                return prompt
        return prompts[0]

    def raw_path(self, mode: str, prompt_id: str, take_id: str | None = None) -> Path:
        """Return a raw WAV path for a new take."""

        _validate_mode(mode)
        ident = take_id or uuid.uuid4().hex[:12]
        return self.base_dir / mode / "raw" / f"{prompt_id}_{ident}.wav"

    def register_take(self, mode: str, prompt: IdentityPrompt, wav_path: Path) -> IdentityTakeMetadata:
        """Analyze and catalog one take without requiring pitch targets."""

        self.ensure_structure()
        quality = analyze_identity_wav(wav_path)
        status = "accepted" if quality["valid"] else "rejected"
        target_dir = self.base_dir / mode / status
        target_dir.mkdir(parents=True, exist_ok=True)
        take_id = uuid.uuid4().hex[:12]
        target = target_dir / f"{prompt.id}_{take_id}.wav"
        if wav_path.resolve() != target.resolve():
            shutil.copy2(wav_path, target)
        metadata = IdentityTakeMetadata(
            id=take_id,
            profile=self.profile,
            mode=mode,
            prompt_id=prompt.id,
            text=prompt.text,
            language=prompt.language,
            wav_path=str(target),
            duration_sec=quality["duration_sec"],
            sample_rate=quality["sample_rate"],
            channels=quality["channels"],
            sample_width_bits=quality["sample_width_bits"],
            rms=quality["rms"],
            peak=quality["peak"],
            clipping=quality["clipping"],
            measured_f0_hz=quality["measured_f0_hz"],
            measured_midi=quality["measured_midi"],
            voiced_ratio=quality["voiced_ratio"],
            status=status,
            reason=quality["reason"],
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self.catalog_path.open("a", encoding="utf-8") as catalog:
            catalog.write(json.dumps(asdict(metadata), ensure_ascii=False) + "\n")
        return metadata

    def takes(self, mode: str | None = None, status: str | None = None) -> list[IdentityTakeMetadata]:
        """Return cataloged takes, optionally filtered."""

        self.ensure_structure()
        rows = []
        for line in self.catalog_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            take = IdentityTakeMetadata(**json.loads(line))
            if mode is not None and take.mode != mode:
                continue
            if status is not None and take.status != status:
                continue
            rows.append(take)
        return rows

    def summary(self) -> dict[str, object]:
        """Return dataset readiness summary."""

        self.ensure_structure()
        return {
            "base_dir": str(self.base_dir),
            "speech_accepted": len(self.takes("speech", "accepted")),
            "singing_accepted": len(self.takes("singing", "accepted")),
            "speech_prompts": len(self.prompts("speech")),
            "singing_prompts": len(self.prompts("singing")),
            "ready_for_first_speech_training": len(self.takes("speech", "accepted")) >= 20,
            "ready_for_first_singing_training": len(self.takes("singing", "accepted")) >= 20,
        }


def analyze_identity_wav(path: Path) -> dict[str, object]:
    """Validate one identity take by quality, not by pitch target."""

    try:
        with wave.open(str(path), "rb") as reader:
            channels = reader.getnchannels()
            width = reader.getsampwidth()
            rate = reader.getframerate()
            frames = reader.getnframes()
            raw = reader.readframes(frames)
    except wave.Error:
        return _invalid("wav invalido")
    if channels != TARGET_CHANNELS:
        return _invalid("debe ser mono")
    if width != TARGET_SAMPLE_WIDTH:
        return _invalid("debe ser PCM 16-bit")
    duration = frames / float(rate) if rate else 0.0
    samples = _samples(raw, channels)
    rms_value, peak_value = _rms_peak(samples)
    f0 = _estimate_f0(samples, rate)
    voiced_ratio = 1.0 if f0 > 0 else 0.0
    reason = None
    valid = True
    if rate != TARGET_SAMPLE_RATE:
        valid, reason = False, "debe estar a 48 kHz"
    elif duration < 1.0:
        valid, reason = False, "duracion insuficiente"
    elif rms_value < 250:
        valid, reason = False, "nivel demasiado bajo"
    elif peak_value >= 32760:
        valid, reason = False, "clipping"
    elif f0 <= 0:
        valid, reason = False, "F0 no detectable"
    return {
        "valid": valid,
        "reason": reason,
        "duration_sec": round(duration, 3),
        "sample_rate": rate,
        "channels": channels,
        "sample_width_bits": width * 8,
        "rms": rms_value,
        "peak": peak_value,
        "clipping": peak_value >= 32760,
        "measured_f0_hz": round(f0, 2),
        "measured_midi": round(_hz_to_midi(f0), 2) if f0 > 0 else 0.0,
        "voiced_ratio": voiced_ratio,
    }


def _validate_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"Modo invalido: {mode}")


def _invalid(reason: str) -> dict[str, object]:
    return {
        "valid": False,
        "reason": reason,
        "duration_sec": 0.0,
        "sample_rate": 0,
        "channels": 0,
        "sample_width_bits": 0,
        "rms": 0,
        "peak": 0,
        "clipping": False,
        "measured_f0_hz": 0.0,
        "measured_midi": 0.0,
        "voiced_ratio": 0.0,
    }


def _samples(raw: bytes, channels: int) -> list[int]:
    return [
        int.from_bytes(raw[index : index + 2], "little", signed=True)
        for index in range(0, len(raw), 2 * channels)
    ]


def _rms_peak(samples: list[int]) -> tuple[int, int]:
    if not samples:
        return 0, 0
    rms_value = int((sum(sample * sample for sample in samples) / len(samples)) ** 0.5)
    return rms_value, max(abs(sample) for sample in samples)


def _estimate_f0(samples: list[int], sample_rate: int) -> float:
    if sample_rate <= 0 or len(samples) < sample_rate // 2:
        return 0.0
    window = samples[: min(len(samples), sample_rate * 3)]
    if max(abs(sample) for sample in window) < 500:
        return 0.0
    crossings = 0
    previous = window[0]
    for sample in window[1:]:
        if (previous <= 0 < sample) or (previous >= 0 > sample):
            crossings += 1
        previous = sample
    duration = len(window) / float(sample_rate)
    estimate = crossings / (2.0 * duration) if duration > 0 else 0.0
    return estimate if 60.0 <= estimate <= 500.0 else 0.0


def _hz_to_midi(hz: float) -> float:
    return 69.0 + 12.0 * math.log2(hz / 440.0)
