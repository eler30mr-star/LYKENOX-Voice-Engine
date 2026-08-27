import subprocess
from pathlib import Path

def main():
    root = Path("D:/Proyectos/LYKENOX-Voice-Engine")
    source = root / "tools" / "renderers" / "utau_basic_bridge" / "LykenoxUtauBridge.cs"
    output = root / "tools" / "renderers" / "utau_basic_bridge" / "lykenox_utau_bridge.exe"

    csc = Path("C:/Windows/Microsoft.NET/Framework64/v4.0.30319/csc.exe")
    if not csc.exists():
        print("Error: csc.exe not found.")
        return

    print(f"Compiling LYKENOX UTAU Bridge from {source}...")
    cmd = [str(csc), "/out:" + str(output), str(source)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"Success! Created {output}")
        else:
            print(f"Failed to compile:\n{res.stdout}\n{res.stderr}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    main()
