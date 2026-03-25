import cv2
import os
import csv
import webbrowser
import threading
from flask import Flask, render_template, Response, jsonify, send_file
from tracker import CentroidTracker
from detector import detect_faces
from datetime import datetime

zone_counts = [0,0,0,0]
app = Flask(__name__)

tracker = CentroidTracker()

entry_count = 0
exit_count = 0
people_inside = 0
peak_count = 0

previous_positions = {}

zones = [
    (100,200,250,350),
    (300,200,450,350),
    (500,200,650,350),
    (700,200,850,350)
]

cap = cv2.VideoCapture(0)

heatmap = None
last_log_time = 0
def log_data():

    file_path = "crowd_data.csv"

    now = datetime.now().strftime("%H:%M:%S")

    row = [
        now,
        people_inside,
        entry_count,
        exit_count,
        zone_counts[0],
        zone_counts[1],
        zone_counts[2],
        zone_counts[3]
    ]

    file_exists = os.path.exists(file_path)

    with open(file_path,"a",newline="") as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["Time","People Inside","Entries","Exits","Zone1","Zone2","Zone3","Zone4"])

        writer.writerow(row)
def generate_frames():

    global entry_count, exit_count, people_inside, peak_count, heatmap

    while True:

        success, frame = cap.read()

        if not success:
            break

        h,w,_ = frame.shape

        if heatmap is None:
            heatmap = cv2.GaussianBlur(frame,(51,51),0)

        center_line = w // 2

        cv2.line(frame,(center_line,0),(center_line,h),(0,0,255),3)

        faces = detect_faces(frame)

        objects = tracker.update(faces)

        for (objectID, centroid) in objects.items():

            cx,cy = centroid

            for (x1,y1,x2,y2) in faces:

                fx=int((x1+x2)/2)
                fy=int((y1+y2)/2)

                if abs(cx-fx)<20 and abs(cy-fy)<20:

                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                    cv2.putText(frame,f"ID {objectID}",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

                    break

            cv2.circle(frame,(cx,cy),5,(0,255,255),-1)

            if objectID in previous_positions:

                prev_x=previous_positions[objectID]

                if prev_x<center_line and cx>center_line:
                    entry_count+=1
                    people_inside+=1

                elif prev_x>center_line and cx<center_line:
                    exit_count+=1
                    people_inside=max(0,people_inside-1)

            previous_positions[objectID]=cx

        global zone_counts
        zone_counts=[0]*len(zones)

        for i,(x1,y1,x2,y2) in enumerate(zones):

            for (objectID,centroid) in objects.items():

                cx,cy=centroid

                if x1<cx<x2 and y1<cy<y2:
                    zone_counts[i]+=1

            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)

            cv2.putText(frame,f"Zone{i+1}:{zone_counts[i]}",
            (x1,y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

        peak_count=max(peak_count,people_inside)
        global last_log_time

        current_time = datetime.now().timestamp()

        if current_time - last_log_time > 1  and (entry_count > 0 or exit_count > 0):
            log_data()
        last_log_time = current_time

        # statistics panel
        cv2.rectangle(frame,(10,10),(260,150),(0,0,0),-1)

        cv2.putText(frame,f"People Inside: {people_inside}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

        cv2.putText(frame,f"Entries: {entry_count}",
            (20,70),
            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        cv2.putText(frame,f"Exits: {exit_count}",
            (20,100),
            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        cv2.putText(frame,f"Peak: {peak_count}",
            (20,130),
            cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)

        heatmap=cv2.addWeighted(heatmap,0.9,frame,0.1,0)

        ret,buffer=cv2.imencode('.jpg',frame)

        frame=buffer.tobytes()

        yield(b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')


@app.route('/')
def dashboard():
    return render_template("dashboard.html")


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def stats():
    return jsonify({
        "inside":people_inside,
        "entries":entry_count,
        "exits":exit_count,
        "zones":zone_counts
    })
@app.route('/alerts')
def alerts():

    path = os.path.join(os.getcwd(), "alerts", "screenshots")

    # ensure folder exists
    os.makedirs(path, exist_ok=True)

    files = []

    if os.path.isdir(path):
        files = sorted(os.listdir(path), reverse=True)[:6]

    return jsonify(files)

@app.route('/logs')
def logs():

    logfile="logs/system_logs.txt"

    lines=[]

    if os.path.exists(logfile):

        with open(logfile) as f:
            lines=f.readlines()[-10:]

    return jsonify(lines)


@app.route('/download_report')
def download_report():
    
    file_path = "crowd_data.csv"

    # create file if not present
    if not os.path.exists(file_path):

        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Time","People Inside","Entries","Exits","Zone1","Zone2","Zone3","Zone4"])
        
    return send_file(file_path, as_attachment=True)
    

def open_browser():
    webbrowser.open("http://127.0.0.1:5000/")


if __name__=="__main__":

    threading.Timer(1,open_browser).start()

    app.run(debug=False)