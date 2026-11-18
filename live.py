"""JAILER live: webcam (or video file) -> fail-closed vision firewall, side-by-side.

  python live.py                     # webcam 0, live window (q to quit)
  python live.py --cam 1             # another camera
  python live.py --video clip.mov    # process a recorded clip -> out/clip_jailed.mp4

Fail closed: whenever the scene changes (or no verdict yet / jailer failed), the WHOLE frame
is pixelated until JAILER has verified the new scene. Then only injection boxes stay pixelated.
"""
import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import jailer_core as jc

TASK_DEFAULT = "Hand the cup to the person"


def thumb(frame):
    return cv2.GaussianBlur(cv2.resize(frame, (64, 48), interpolation=cv2.INTER_AREA), (3, 3), 0).astype(np.float32)


def changed(a, b, thresh):
    """True if more than `thresh` % of thumbnail pixels changed by > 30 in any color channel."""
    return a is None or b is None or float(np.mean(np.abs(a - b).max(axis=2) > 30)) * 100 > thresh


def banner(img, text, color):
    img = img.copy()
    cv2.rectangle(img, (0, img.shape[0] - 44), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    cv2.putText(img, text[:90], (10, img.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return img


def header(img, text, color):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return img


class Verdict:
    """Latest jailer verdict + the thumbnail of the frame it was computed on."""
    def __init__(self):
        self.lock = threading.Lock()
        self.boxes, self.thumb, self.summary, self.ok, self.ts = [], None, "", False, 0.0

    def set(self, boxes, th, summary, ok):
        with self.lock:
            self.boxes, self.thumb, self.summary, self.ok, self.ts = boxes, th, summary, ok, time.time()

    def get(self):
        with self.lock:
            return list(self.boxes), self.thumb, self.summary, self.ok


def judge(frame, task):
    """Run jailer on one frame -> (boxes, summary, ok). Any failure => ok=False (fail closed)."""
    try:
        jr = jc.run_jailer(frame, task)
        boxes = [i["box"] for i in jr["items"] if i["verdict"] == "injection"]
        texts = [i["text"] for i in jr["items"] if i["verdict"] == "injection"]
        msg = f"JAILED {len(boxes)} injection(s) - robot continues its task" if boxes else "Scene verified: no injections"
        print("[jailer]", texts)
        return boxes, msg, True
    except Exception as e:
        return [], f"JAILER FAILED ({type(e).__name__}) - HALT", False


def render(frame, verdict_boxes, verdict_thumb, ok, summary, thresh, pending, strict=True):
    """strict: any scene change blurs the whole frame until re-verified (max safety).
    non-strict: only a missing/failed verdict blurs the whole frame; otherwise the latest
    jailed boxes are applied while JAILER keeps re-checking in the background."""
    th = thumb(frame)
    if not ok or (strict and changed(th, verdict_thumb, thresh)):
        safe = jc.fail_closed_frame(frame)
        msg = "SCENE CHANGED - VERIFYING..." if pending else (summary if not ok else "UNVERIFIED")
        return banner(safe, msg, (0, 200, 255)), False
    safe = jc.jail_boxes(frame, verdict_boxes) if verdict_boxes else frame
    color = (0, 0, 255) if verdict_boxes else (80, 220, 80)
    if pending and not strict:
        summary = summary + "  (re-checking...)"
    return banner(safe, summary, color), True


def side(raw, safe):
    gap = np.full((raw.shape[0], 10, 3), 255, np.uint8)
    return np.hstack([header(raw, "RAW CAMERA", (80, 80, 255)), gap,
                      header(safe, "WHAT THE ROBOT SEES (JAILER)", (80, 220, 80))])


def live(cam, task, thresh, width, strict=False):
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {cam}")
    v = Verdict()
    latest = {"frame": None}
    busy = threading.Event()
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            f = latest["frame"]
            _, vth, _, ok = v.get()
            # strict: re-check only on scene change; default: re-check back-to-back, always
            if f is None or (strict and ok and not changed(thumb(f), vth, thresh)):
                time.sleep(0.1)
                continue
            # wait for the scene to settle a moment so we judge a steady frame
            time.sleep(0.4 if strict else 0.05)
            f = latest["frame"].copy()
            busy.set()
            boxes, msg, ok = judge(f, task)
            v.set(boxes, thumb(f), msg, ok)
            busy.clear()
            if not ok:
                time.sleep(2)

    threading.Thread(target=worker, daemon=True).start()
    cv2.namedWindow("JAILER live", cv2.WINDOW_NORMAL)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if width and frame.shape[1] != width:
            frame = cv2.resize(frame, (width, int(frame.shape[0] * width / frame.shape[1])))
        latest["frame"] = frame
        boxes, vth, summary, vok = v.get()
        safe, _ = render(frame, boxes, vth, vok, summary, thresh, busy.is_set(), strict)
        cv2.imshow("JAILER live", side(frame, safe))
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    stop.set()
    cap.release()
    cv2.destroyAllWindows()


def video(path, task, thresh, width, every):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    jc.OUT_DIR.mkdir(exist_ok=True)
    stem = Path(path).stem
    writers = {}
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if width and frame.shape[1] != width:
            frame = cv2.resize(frame, (width, int(frame.shape[0] * width / frame.shape[1])))
        frames.append(frame)
    # offline: verify a keyframe every `every` seconds (in parallel), BEFORE rendering any frame.
    step = max(1, int(round(every * fps)))
    keys = list(range(0, len(frames), step))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as ex:
        verdicts = dict(zip(keys, ex.map(lambda i: judge(frames[i], task), keys)))
    for k in keys:
        print(f"t={k / fps:6.2f}s  {verdicts[k][1]}")
    for n, frame in enumerate(frames):
        k = keys[n // step]
        boxes, summary, vok = verdicts[k]
        # fail closed if this frame drifted too far from the keyframe that was verified
        safe, _ = render(frame, boxes, thumb(frames[k]), vok, summary, thresh, False)
        sbs = side(frame, safe)
        for name, img in (("sbs", sbs), ("jailed", safe)):
            if name not in writers:
                h, w = img.shape[:2]
                writers[name] = cv2.VideoWriter(str(jc.OUT_DIR / f"{stem}_{name}.mp4"),
                                                cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writers[name].write(img)
    for w in writers.values():
        w.release()
    print("wrote", *[str(jc.OUT_DIR / f"{stem}_{k}.mp4") for k in writers])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="JAILER live / video")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--video")
    ap.add_argument("--task", default=TASK_DEFAULT)
    ap.add_argument("--thresh", type=float, default=3.0,
                    help="scene-change threshold: %% of pixels changed (raise if it flickers)")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--strict", action="store_true",
                    help="live mode: blur the WHOLE frame on any scene change until re-verified")
    ap.add_argument("--every", type=float, default=2.0, help="video mode: re-judge every N seconds")
    a = ap.parse_args()
    if a.video:
        video(a.video, a.task, a.thresh, a.width, a.every)
    else:
        live(a.cam, a.task, a.thresh, a.width, a.strict)
