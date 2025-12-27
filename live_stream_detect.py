import argparse
import sys
import time

import cv2
import numpy as np
from insightface.app import FaceAnalysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtsp", required=True)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--det-size", default="640,640")
    parser.add_argument("--min-score", type=float, default=0.6)
    parser.add_argument("--max-fps", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    det_w, det_h = (int(x.strip()) for x in args.det_size.split(","))

    providers = ["CPUExecutionProvider"]
    if args.device == "gpu":
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

    app = FaceAnalysis(providers=providers)
    app.prepare(ctx_id=-1, det_size=(det_w, det_h))

    cap = cv2.VideoCapture(args.rtsp)
    if not cap.isOpened():
        print(f"Failed to open stream: {args.rtsp}", file=sys.stderr)
        return 2

    last_frame_time = 0.0
    window_name = "Live Face Detection"

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.05)
            continue

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        faces = app.get(frame)

        for face in faces:
            if face.det_score is None or face.det_score < args.min_score:
                continue

            box = face.bbox.astype(np.int32)
            x1, y1, x2, y2 = box.tolist()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"face {float(face.det_score):.2f}"
            cv2.putText(
                frame,
                label,
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        if args.max_fps and args.max_fps > 0:
            now = time.time()
            if last_frame_time > 0:
                elapsed = now - last_frame_time
                target = 1.0 / float(args.max_fps)
                if elapsed < target:
                    time.sleep(target - elapsed)
            last_frame_time = time.time()

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
