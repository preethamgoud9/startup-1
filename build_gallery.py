import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="gallery.npz")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--det-size", default="640,640")
    parser.add_argument("--min-score", type=float, default=0.6)
    parser.add_argument("--strategy", choices=["mean", "all"], default="mean")
    return parser.parse_args()


def list_image_files(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out: list[Path] = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return sorted(out)


def largest_face(faces):
    if not faces:
        return None
    areas = []
    for f in faces:
        x1, y1, x2, y2 = f.bbox
        areas.append(float(max(0.0, x2 - x1) * max(0.0, y2 - y1)))
    return faces[int(np.argmax(areas))]


def main() -> int:
    args = parse_args()
    det_w, det_h = (int(x.strip()) for x in args.det_size.split(","))

    providers = ["CPUExecutionProvider"]
    if args.device == "gpu":
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

    app = FaceAnalysis(providers=providers)
    app.prepare(ctx_id=-1, det_size=(det_w, det_h))

    data_dir = Path(args.data_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        print(f"Invalid --data-dir: {data_dir}", file=sys.stderr)
        return 2

    student_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if not student_dirs:
        print(f"No student subfolders found in: {data_dir}", file=sys.stderr)
        return 2

    labels: list[str] = []
    embeddings: list[np.ndarray] = []

    for student_dir in student_dirs:
        student_id = student_dir.name
        image_files = list_image_files(student_dir)
        if not image_files:
            print(f"Skipping {student_id}: no images")
            continue

        student_embs: list[np.ndarray] = []
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            faces = app.get(img)
            face = largest_face([f for f in faces if f.det_score is not None and f.det_score >= args.min_score])
            if face is None:
                continue

            emb = getattr(face, "normed_embedding", None)
            if emb is None:
                continue

            student_embs.append(np.asarray(emb, dtype=np.float32))

        if not student_embs:
            print(f"Skipping {student_id}: no usable faces")
            continue

        if args.strategy == "mean":
            e = np.mean(np.stack(student_embs, axis=0), axis=0)
            e = e / (np.linalg.norm(e) + 1e-12)
            labels.append(student_id)
            embeddings.append(e.astype(np.float32))
        else:
            for e in student_embs:
                labels.append(student_id)
                embeddings.append(e.astype(np.float32))

        print(f"Enrolled {student_id}: {len(student_embs)} images")

    if not embeddings:
        print("No embeddings generated.", file=sys.stderr)
        return 2

    emb_arr = np.stack(embeddings, axis=0)
    label_arr = np.array(labels, dtype=object)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), embeddings=emb_arr, labels=label_arr)

    print(f"Saved gallery: {out_path} ({emb_arr.shape[0]} embeddings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
