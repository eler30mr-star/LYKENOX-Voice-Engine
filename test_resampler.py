import wave
import struct
from pathlib import Path
from lykenox_voice_engine.core.voicebank import UtauEngine, OtoEntry

def create_oto_entry(alias, wav_name):
    return OtoEntry(
        alias=alias,
        wav=wav_name,
        offset=0,
        consonant=100,
        cutoff=-100,
        preutterance=50,
        overlap=25
    )

def main():
    source_wav = Path("tools/nnsvs_env/nnsvs_source/tests/data/nitech_jp_song070_f001_004.wav")
    if not source_wav.exists():
        print(f"Error: No se encuentra {source_wav}")
        return

    wav_dir = source_wav.parent
    oto = {"test": create_oto_entry("test", source_wav.name)}

    # Mock alias resolver
    def resolver(lyric):
        return ["test"]

    engine = UtauEngine(wav_dir, oto, 48000, resolver)

    # Test cases
    from lykenox_voice_engine.models.notes import NoteEvent

    print("Testing pitch shift 60 (C4)...")
    note60 = [NoteEvent("test", 60, 0.0, 1.0)]
    out60 = engine.render(note60)

    print("Testing pitch shift 64 (E4)...")
    note64 = [NoteEvent("test", 64, 0.0, 1.0)]
    out64 = engine.render(note64)

    print("Testing pitch shift 57 (A3)...")
    note57 = [NoteEvent("test", 57, 0.0, 1.0)]
    out57 = engine.render(note57)

    # Save results
    output_dir = Path("outputs/test_resampler")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("c4", out60), ("e4", out64), ("a3", out57)]:
        with wave.open(str(output_dir / f"{name}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(48000)
            w.writeframes(data)

    print(f"Resultados guardados en {output_dir}")

if __name__ == "__main__":
    main()
