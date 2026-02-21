"""Downloads the antelopev2 face recognition model via InsightFace's built-in downloader."""
import sys

def download():
    from insightface.app import FaceAnalysis

    try:
        print("[INFO] Downloading antelopev2 (~344MB, one-time)...")
        app = FaceAnalysis(name="antelopev2", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        print("[INFO] antelopev2 model downloaded successfully.")
    except Exception as e:
        print(f"[WARN] antelopev2 unavailable ({e}), falling back to buffalo_l...")
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        print("[INFO] buffalo_l model downloaded successfully.")

if __name__ == "__main__":
    download()
