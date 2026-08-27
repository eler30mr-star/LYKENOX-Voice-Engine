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

    # 1. Internal Renderer
    start = time.perf_counter()
    vocal_internal = output_dir / "vocal_internal_renderer.wav"
    engine.synthesize_to_path("lykenox", "baila conmigo", notes, 120, vocal_internal, renderer="internal")
    end = time.perf_counter()
    print(f"Renderer Interno: {vocal_internal} ({end - start:.2f}s)")

    # 2. Classic Renderer (External)
    # Check if a resampler is present
    from lykenox_voice_engine.core.resampler_interface import UtauClassicRenderer
    classic = UtauClassicRenderer(root)

    if classic.resampler:
        print(f"Resampler detectado: {classic.resampler.name}")
        start = time.perf_counter()
        vocal_classic = output_dir / "vocal_openutau_renderer.wav"
        engine.synthesize_to_path("lykenox", "baila conmigo", notes, 120, vocal_classic, renderer="classic")
        end = time.perf_counter()
        print(f"Renderer Clásico: {vocal_classic} ({end - start:.2f}s)")
    else:
        print("Renderer Clásico: NO DISPONIBLE (resampler.exe no encontrado en tools/renderers/)")
        print("Para probarlo, copie 'resampler.exe' o 'tips.exe' a 'D:/Proyectos/LYKENOX-Voice-Engine/tools/renderers/'")

if __name__ == "__main__":
    main()
