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
]
