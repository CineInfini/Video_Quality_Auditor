"""Download videos from URLs (simple version)"""
import requests
import zipfile
import cv2
from pathlib import Path

def download_video(url, name, dest_dir, is_zip=False, retries=2):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    video_file = dest_dir / f"{name}.mp4"
    if video_file.exists():
        cap = cv2.VideoCapture(str(video_file))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if frames > 0:
            print(f"✅ {name} already valid ({frames} frames)")
            return video_file
        else:
            print(f"⚠️ Existing {name} is invalid (0 frames). Re-downloading.")
            video_file.unlink()
    for attempt in range(retries):
        try:
            print(f"📥 Downloading {name} (attempt {attempt+1})...")
            if is_zip:
                zip_path = dest_dir / f"{name}.zip"
                r = requests.get(url, stream=True, headers={"User-Agent": "Mozilla/5.0"})
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(dest_dir)
                extracted = list(dest_dir.glob("*.mp4")) + list(dest_dir.glob("*.mkv")) + list(dest_dir.glob("*.mov"))
                if extracted:
                    extracted[0].rename(video_file)
                zip_path.unlink()
            else:
                r = requests.get(url, stream=True, headers={"User-Agent": "Mozilla/5.0"})
                with open(video_file, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            cap = cv2.VideoCapture(str(video_file))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if frames > 0:
                print(f"  ✅ {name} ready ({frames} frames)")
                return video_file
            else:
                raise Exception("Downloaded file has 0 frames")
        except Exception as e:
            print(f"  ❌ Attempt {attempt+1} failed: {e}")
            if video_file.exists():
                video_file.unlink()
            if attempt == retries-1:
                print(f"  → Giving up on {name}")
                return None
    return None
