import shutil
from pathlib import Path
from lykenox_voice_engine.engines.utau_engine import UtauSampleEngine
from lykenox_voice_engine.models.notes import NoteEvent

def main():
    root = Path("D:/Proyectos/LYKENOX-Voice-Engine")
    engine = UtauSampleEngine(root)

    # "baila conmigo" score
    notes = [
        NoteEvent("bai", 60, 0.0, 0.45),
        NoteEvent("la", 62, 0.45, 0.45),
        NoteEvent("con", 64, 0.9, 0.45),
        NoteEvent("mi", 62, 1.35, 0.45),
        NoteEvent("go", 60, 1.8, 0.6),
    ]

    output_dir = root / "outputs" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    new_vocal = output_dir / "vocal_new.wav"

    print("Sintetizando 'baila conmigo' con el nuevo renderer...")
    try:
        engine.synthesize_to_path("lykenox", "baila conmigo", notes, 120, new_vocal)
        print(f"Nuevo vocal generado en: {new_vocal}")
    except Exception as e:
        print(f"Error al sintetizar: {e}")

if __name__ == "__main__":
    main()
