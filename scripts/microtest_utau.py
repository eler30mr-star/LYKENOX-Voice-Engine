"""Microtest for the Spanish Lite sample-based backend."""

from pathlib import Path
from lykenox_voice_engine.engines.utau_engine import UtauSampleEngine
from lykenox_voice_engine.models.notes import NoteEvent

def run_microtest():
    root = Path(__file__).resolve().parents[1]
    engine = UtauSampleEngine(root)
    profile = "lykenox"

    print("--- LYKENOX Voice Engine: Microtest UTAU ---")

    # 1. Check availability
    status = engine.check_available()
    print(f"Backend: {status['backend']}")
    print(f"Voicebank Coverage: {status['voicebank_coverage']}%")

    # 2. Mock score "baila"
    lyrics = "baila"
    notes = [
        NoteEvent("bai", 60, 0.0, 0.5),
        NoteEvent("la", 62, 0.5, 0.5),
    ]
    tempo = 120

    print(f"Intentando sintetizar: '{lyrics}'...")

    try:
        output = engine.synthesize(profile, lyrics, notes, tempo)
        print(f"OK! WAV generado en: {output}")
    except RuntimeError as e:
        print(f"FAIL: {e}")
        print("Nota: Es normal si aun no has grabado los samples 'bai' y 'la'.")

if __name__ == "__main__":
    run_microtest()
