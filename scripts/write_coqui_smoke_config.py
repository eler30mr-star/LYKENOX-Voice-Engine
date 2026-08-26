"""Write a minimal Coqui VITS config for CPU smoke training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    """Generate a Coqui config using the installed TTS package."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "datasets" / "lykenox" / "identity_voice" / "prepared" / "coqui_segmented",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "models" / "lykenox_identity" / "coqui_smoke" / "config.json",
    )
    args = parser.parse_args()

    from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
    from TTS.tts.configs.vits_config import VitsConfig

    output_dir = args.output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    config = VitsConfig(
        run_name="lykenox_coqui_vits_smoke",
        output_path=str(output_dir),
        datasets=[
            BaseDatasetConfig(
                formatter="coqui",
                dataset_name="lykenox_segmented",
                path=str(args.dataset_dir),
                meta_file_train="metadata_train.csv",
                meta_file_val="metadata_val.csv",
                language="es",
            )
        ],
        batch_size=1,
        eval_batch_size=1,
        num_loader_workers=0,
        num_eval_loader_workers=0,
        epochs=1,
        print_step=1,
        save_step=25,
        save_n_checkpoints=1,
        run_eval=True,
        test_delay_epochs=-1,
        use_phonemes=False,
        phoneme_language="es-es",
        text_cleaner="multilingual_cleaners",
        characters=CharactersConfig(
            characters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÁÉÍÓÚÜÑáéíóúüñ",
            punctuations="!'(),-.:;?¿¡ ",
        ),
        min_text_len=10,
        max_text_len=280,
        max_audio_len=48_000 * 18,
        start_by_longest=False,
        test_sentences=[["Hola, esta es una prueba de mi voz LYKENOX."]],
    )
    config.audio.sample_rate = 48_000
    config.audio.resample = False
    config.audio.mel_fmin = 40
    config.audio.mel_fmax = 12_000
    config.save_json(str(args.output_path))
    print(args.output_path)


if __name__ == "__main__":
    main()
