import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import csv
import os
import time
from tracker import CentroidTracker
from datetime import datetime
from detector import detect_faces

MAX_CAPACITY = 20

tracker = CentroidTracker()

zones = []
drawing = False
start_point = None

entry_count = 0
exit_count = 0
people_inside = 0
peak_count = 0

previous_positions = {}
crossed_ids = set()

frame_count = 0
SAVE_INTERVAL = 150

last_alert_time = 0
ALERT_COOLDOWN = 10


def log_event(msg):
    with open("logs/system_logs.txt", "a") as f:
        f.write(f"{datetime.now()} - {msg}\n")


def save_data(zone_name, count):

    file_exists = os.path.isfile("crowd_data.csv")

    with open("crowd_data.csv", "a", newline="") as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["time", "zone", "count"])

        writer.writerow([
            datetime.now().strftime("%H:%M:%S"),
            zone_name,
            count
        ])


def mouse_callback(event, x, y, flags, param):

    global drawing, start_point, zones

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        zones.append((start_point[0], start_point[1], x, y))

    elif event == cv2.EVENT_RBUTTONDOWN:
        if zones:
            zones.pop()


cap = cv2.VideoCapture(0)

cv2.namedWindow("Crowd Monitoring")
cv2.setMouseCallback("Crowd Monitoring", mouse_callback)

log_event("Camera started")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    height, width, _ = frame.shape
    center_line = width // 2

    # draw entry/exit divider
    cv2.line(frame, (center_line, 0), (center_line, height), (0, 0, 255), 3)

    cv2.putText(frame, "ENTRY", (50, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, "EXIT", (width - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    faces = detect_faces(frame)

    # ---------- TRACKER LOGIC START ----------
    objects = tracker.update(faces)

    for (objectID, centroid) in objects.items():

        cx, cy = centroid

        # find corresponding face box
        for (x1, y1, x2, y2) in faces:

            fx = int((x1 + x2) / 2)
            fy = int((y1 + y2) / 2)

            if abs(cx - fx) < 20 and abs(cy - fy) < 20:

                # draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # draw ID
                cv2.putText(frame, f"ID {objectID}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

                break

        # draw centroid
        cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

        # entry exit logic
        if objectID in previous_positions:

            prev_x = previous_positions[objectID]

            if prev_x < center_line and cx > center_line and objectID not in crossed_ids:

                entry_count += 1
                people_inside += 1
                crossed_ids.add(objectID)

                log_event(f"Entry detected ID {objectID}")

            elif prev_x > center_line and cx < center_line and objectID in crossed_ids:

                exit_count += 1
                people_inside = max(0, people_inside - 1)
                crossed_ids.remove(objectID)

                log_event(f"Exit detected ID {objectID}")

        previous_positions[objectID] = cx

    # ---------- TRACKER LOGIC END ----------

    # -------- ZONE COUNTING (using tracker IDs) --------
    zone_counts = [0] * len(zones)

    for i, zone in enumerate(zones):

        x1, y1, x2, y2 = zone

        for (objectID, centroid) in objects.items():

            cx, cy = centroid

            if x1 < cx < x2 and y1 < cy < y2:
                zone_counts[i] += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.putText(frame, f"Zone {i+1}: {zone_counts[i]}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # -------- SAVE DATA WITH INTERVAL --------
    for i, count in enumerate(zone_counts):

        if frame_count % SAVE_INTERVAL == 0:
            save_data(f"Zone{i+1}", count)

        current_time = time.time()

        if count > MAX_CAPACITY:

            cv2.putText(frame, "OVER CROWD ALERT",
                        (100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)

            if current_time - last_alert_time > ALERT_COOLDOWN:

                filename = f"alerts/screenshots/alert_{datetime.now().strftime('%H%M%S')}.jpg"

                cv2.imwrite(filename, frame)

                log_event(f"Alert triggered in Zone {i+1}")

                last_alert_time = current_time

    peak_count = max(peak_count, people_inside)

    # -------- STATISTICS PANEL --------
    cv2.rectangle(frame, (10, 10), (260, 150), (0, 0, 0), -1)

    cv2.putText(frame, f"People Inside: {people_inside}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(frame, f"Entries: {entry_count}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"Exits: {exit_count}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, f"Peak: {peak_count}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Crowd Monitoring", frame)

    key = cv2.waitKey(1)

    if key == ord('r'):
        zones.clear()

    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

log_event("System shutdown")