import sys
from pathlib import Path
from lykenox_voice_engine.core.voicebank import VoicebankManager

def main():
    root = Path("D:/Proyectos/LYKENOX-Voice-Engine")
    manager = VoicebankManager(root, profile="lykenox")

    print("Iniciando prueba de renderizado UTAU...")

    # Intentar renderizar la vocal 'a' en los tonos solicitados: 60 (C4), 64 (E4), 57 (A3)
    try:
        output_path = manager.test_renderer(alias="a")
        print(f"Prueba completada con éxito. Audio generado en: {output_path}")
    except Exception as e:
        print(f"Error durante la prueba: {e}")
        print("Asegúrate de que el voicebank esté construido y contenga el alias 'a'.")

if __name__ == "__main__":
    main()
