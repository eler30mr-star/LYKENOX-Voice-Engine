"""Build LYKENOX speech mel caches from the engine-neutral manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lykenox_voice_engine.models.speech import LykenoxSpeechConfig
from lykenox_voice_engine.training.speech_dataset import LykenoxSpeechDataset


def build_cache(root: Path, split: str = "train") -> dict[str, object]:
    prepared = root / "datasets" / "lykenox" / "identity_voice" / "prepared" / "speech_segmented"
    csv_path = prepared / ("train.segmented.csv" if split == "train" else "val.segmented.csv")
    if not csv_path.exists():
        fallback = root / "datasets" / "lykenox" / "identity_voice" / "prepared" / "speech" / f"{split}.csv"
        csv_path = fallback
    if not csv_path.exists():
        raise FileNotFoundError(f"No LYKENOX speech manifest found for {split}: {csv_path}")

    cache_dir = root / "datasets" / "lykenox" / "identity_voice" / "features" / "speech" / "mel-v1" / split
    dataset = LykenoxSpeechDataset(csv_path, cache_dir, LykenoxSpeechConfig())
    frames = 0
    for index in range(len(dataset)):
        item = dataset[index]
        frames += int(item["mel"].shape[0])

    return {
        "status": "pass",
        "split": split,
        "manifest": str(csv_path),
        "cache_dir": str(cache_dir),
        "utterances": len(dataset),
        "mel_frames": frames,
        "note": "Feature cache generated from the LYKENOX master-derived manifest; no third-party trainer format is canonical.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--split", choices=("train", "val"), default="train")
    args = parser.parse_args()
    print(json.dumps(build_cache(args.root.resolve(), args.split), indent=2))


if __name__ == "__main__":
    main()
