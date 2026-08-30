"""LYKENOX-owned neural vocoder components."""

from .config import LykenoxVocoderConfig
from .discriminator import (
    DISCRIMINATOR_ARCHITECTURE,
    DiscriminatorOutput,
    LykenoxMultiScaleWaveformDiscriminator,
)
from .network import LykenoxVocoderGenerator
from .network_v1 import (
    VOCODER_GENERATOR_V1_ARCHITECTURE,
    LykenoxVocoderGeneratorV1,
)
from .network_v2 import (
    VOCODER_GENERATOR_V2_ARCHITECTURE,
    LykenoxVocoderGeneratorV2,
)
from .network_v3 import (
    VOCODER_GENERATOR_V3_ARCHITECTURE,
    LykenoxVocoderGeneratorV3,
)
from .network_v4 import (
    VOCODER_GENERATOR_V4_ARCHITECTURE,
    LykenoxVocoderGeneratorV4,
)
from .network_v4_1 import (
    VOCODER_GENERATOR_V4_1_ARCHITECTURE,
    LykenoxVocoderGeneratorV41,
)
from .network_v4_2 import (
    VOCODER_GENERATOR_V4_2_ARCHITECTURE,
    LykenoxVocoderGeneratorV42,
)
from .network_v4_3 import (
    VOCODER_GENERATOR_V4_3_ARCHITECTURE,
    LykenoxVocoderGeneratorV43,
)
from .network_v4_4 import (
    VOCODER_GENERATOR_V4_4_ARCHITECTURE,
    LykenoxVocoderGeneratorV44,
)
from .network_v5 import (
    VOCODER_GENERATOR_V5_ARCHITECTURE,
    LykenoxVocoderGeneratorV5,
)
from .network_v6 import (
    VOCODER_GENERATOR_V6_ARCHITECTURE,
    LykenoxVocoderGeneratorV6,
)

# The legacy flags on network_v6 describe only whether a waveform is directly bypassed.
# They were too narrow to justify the former "source-free" claim: V6 injects accumulated
# F0 phase, a periodic phase aperture, and deterministic unvoiced noise at sample rate.
# Keep the model/checkpoint class loadable for forensics, but expose the honest semantic
# and perceptual status on the public model surface.
LykenoxVocoderGeneratorV6.source_free = False
LykenoxVocoderGeneratorV6.sample_phase_conditioning = True
LykenoxVocoderGeneratorV6.deterministic_unvoiced_noise_conditioning = True
LykenoxVocoderGeneratorV6.local_unit_rms_shape_normalization = True
LykenoxVocoderGeneratorV6.perceptually_rejected = True
LykenoxVocoderGeneratorV6.rejection_date = "2026-08-30"
LykenoxVocoderGeneratorV6.rejection_gate = "full_utterance_oracle_listening"

__all__ = [
    "DISCRIMINATOR_ARCHITECTURE",
    "DiscriminatorOutput",
    "LykenoxMultiScaleWaveformDiscriminator",
    "LykenoxVocoderConfig",
    "LykenoxVocoderGenerator",
    "VOCODER_GENERATOR_V1_ARCHITECTURE",
    "LykenoxVocoderGeneratorV1",
    "VOCODER_GENERATOR_V2_ARCHITECTURE",
    "LykenoxVocoderGeneratorV2",
    "VOCODER_GENERATOR_V3_ARCHITECTURE",
    "LykenoxVocoderGeneratorV3",
    "VOCODER_GENERATOR_V4_ARCHITECTURE",
    "LykenoxVocoderGeneratorV4",
    "VOCODER_GENERATOR_V4_1_ARCHITECTURE",
    "LykenoxVocoderGeneratorV41",
    "VOCODER_GENERATOR_V4_2_ARCHITECTURE",
    "LykenoxVocoderGeneratorV42",
    "VOCODER_GENERATOR_V4_3_ARCHITECTURE",
    "LykenoxVocoderGeneratorV43",
    "VOCODER_GENERATOR_V4_4_ARCHITECTURE",
    "LykenoxVocoderGeneratorV44",
    "VOCODER_GENERATOR_V5_ARCHITECTURE",
    "LykenoxVocoderGeneratorV5",
    "VOCODER_GENERATOR_V6_ARCHITECTURE",
    "LykenoxVocoderGeneratorV6",
]
