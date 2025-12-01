import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # helps avoid libiomp errors on macOS

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

# ============= USER SETTINGS ============= #
VIDEO_SOURCE = "cctv.mp4"   # 👉 put your video file name here, or 0 for webcam
CONF_THRESHOLD = 0.5        # minimum confidence for detections
TRACK_ONLY_PERSON = False   # True = track only "person" class
# ======================================== #


def main():
    # 1. Load YOLO model (downloads yolov8n.pt automatically if not present)
    print("[INFO] Loading YOLO model...")
    model = YOLO("yolov8n.pt")
    class_names = model.names  # COCO classes

    # 2. Initialize DeepSORT tracker
    print("[INFO] Initializing DeepSORT tracker...")
    tracker = DeepSort(
        max_age=30,              # frames to keep track without detections
        n_init=3,                # detections before track is confirmed
        max_cosine_distance=0.3  # appearance distance threshold
        # other params can be added if needed
    )

    # 3. Open video source
    print(f"[INFO] Opening video source: {VIDEO_SOURCE}")
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {VIDEO_SOURCE}")
        return

    # macOS GUI helpers
    cv2.startWindowThread()
    cv2.namedWindow("YOLO + DeepSORT Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO + DeepSORT Tracking", 1280, 720)

    print("[INFO] Tracking started. Press ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video / cannot read frame.")
            break

        # 4. YOLO detection on this frame
        results = model(frame)[0]  # first (and only) result for this frame

        detections = []  # list of ([x, y, w, h], conf, class_id)

        # results.boxes.data: each row = [x1, y1, x2, y2, conf, cls] :contentReference[oaicite:0]{index=0}
        if results.boxes is not None and len(results.boxes) > 0:
            for x1, y1, x2, y2, conf, cls_id in results.boxes.data.tolist():
                conf = float(conf)
                cls_id = int(cls_id)

                if conf < CONF_THRESHOLD:
                    continue

                cls_name = class_names[cls_id]

                # only track persons if option enabled
                if TRACK_ONLY_PERSON and cls_name != "person":
                    continue

                w = x2 - x1
                h = y2 - y1

                # DeepSort expects: ([x, y, w, h], confidence, class_id) :contentReference[oaicite:1]{index=1}
                detections.append(([x1, y1, w, h], conf, cls_id))

        # 5. Update DeepSORT tracker with current frame detections
        tracks = tracker.update_tracks(detections, frame=frame)

        # 6. Draw tracked boxes
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()  # [left, top, right, bottom]
            x1, y1, x2, y2 = map(int, ltrb)

            det_class = getattr(track, "det_class", None)
            if det_class is not None:
                cls_name = class_names[int(det_class)]
                label = f"{cls_name} ID {track_id}"
            else:
                label = f"ID {track_id}"

            # draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # draw label
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # 7. Show frame
        cv2.imshow("YOLO + DeepSORT Tracking", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Tracking finished.")


if __name__ == "__main__":
    main()
