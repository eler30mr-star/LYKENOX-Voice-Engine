import sys
import time
from pathlib import Path
from lykenox_voice_engine.engines.utau_engine import UtauSampleEngine
from lykenox_voice_engine.models.notes import NoteEvent

def main():
    root = Path("D:/Proyectos/LYKENOX-Voice-Engine")
    engine = UtauSampleEngine(root)

    notes = [
        NoteEvent("bai", 60, 0.0, 0.45),
        NoteEvent("la", 62, 0.45, 0.45),
        NoteEvent("con", 64, 0.9, 0.45),
        NoteEvent("mi", 62, 1.35, 0.45),
        NoteEvent("go", 60, 1.8, 0.6),
    ]

    output_dir = root / "outputs" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    vocal_worldline = output_dir / "vocal_worldline.wav"

    print("Sintetizando 'baila conmigo' con LYKENOX UTAU Bridge...")
    try:
        engine.synthesize_to_path("lykenox", "baila conmigo", notes, 120, vocal_worldline, renderer="classic")
        if vocal_worldline.exists() and vocal_worldline.stat().st_size > 44:
            print(f"ÉXITO: vocal_worldline.wav generado en: {vocal_worldline}")
            print(f"Tamaño: {vocal_worldline.stat().st_size} bytes")
        else:
            print("ERROR: El archivo generado está vacío o no existe.")
    except Exception as e:
        print(f"Error al sintetizar: {e}")

if __name__ == "__main__":
    main()
