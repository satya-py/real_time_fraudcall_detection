import subprocess
from pathlib import Path

# 🔧 VLC PATH (verify this exists)
VLC_PATH = r"D:\Program Files\VideoLAN\VLC\vlc.exe"

# 🎯 TARGET SAMPLE RATE
TARGET_SR = 16000

# 🎵 SUPPORTED INPUT FORMATS
SUPPORTED_EXTENSIONS = {".unknown", ".mp3", ".aac", ".mp4", ".m4a", ".wav"}

def convert_with_vlc(input_file: Path, output_file: Path):
    """
    Convert any audio/video to WAV 16kHz mono using VLC (NO FFMPEG)
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        VLC_PATH,
        "--intf", "dummy",
        "--quiet",
        str(input_file),
        "--sout",
        f"#transcode{{acodec=s16l,channels=1,samplerate={TARGET_SR}}}:std{{access=file,mux=wav,dst={output_file}}}",
        "vlc://quit"
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Converted → {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {input_file} | {e}")

def process_dataset(input_root, output_root):
    input_root = Path(input_root)
    output_root = Path(output_root)

    if not input_root.exists():
        print("❌ Input dataset directory does not exist")
        return

    for category_dir in input_root.iterdir():
        if not category_dir.is_dir():
            continue

        category_name = category_dir.name
        print(f"\n📂 Processing category: {category_name}")

        for file in category_dir.rglob("*"):
            if file.suffix.lower() in SUPPORTED_EXTENSIONS:
                relative_path = file.relative_to(input_root)
                output_wav = (output_root / relative_path).with_suffix(".wav")

                # Skip if already processed
                if output_wav.exists():
                    continue

                print(f"🎧 Processing: {file}")
                convert_with_vlc(file, output_wav)

if __name__ == "__main__":
    INPUT_DATASET = r"D:\d_drive_project\Realtime_fraud_call_detection\dataset"
    OUTPUT_DATASET = r"D:\d_drive_project\Realtime_fraud_call_detection\processed_dataset"

    process_dataset(INPUT_DATASET, OUTPUT_DATASET)