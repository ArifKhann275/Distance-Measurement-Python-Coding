
import cv2
import numpy as np
import math
import time
import torch
from ultralytics import YOLO

VIDEO_PATH = "third.mp4"
OUTPUT_PATH = "OUtput 2.mp4"
YOLO_MODEL = "yolov8n.pt"
ASSUMED_HEIGHT_M = 1.70
MIN_CONF = 0.35
IOU_MATCH_THRESH = 0.3
MAX_MISSED = 40
USE_GPU = torch.cuda.is_available()

lk_params = dict(winSize=(15,15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

gftt_params = dict(maxCorners=40, qualityLevel=0.01, minDistance=7, blockSize=7)

def iou(a,b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    interW = max(0, xB-xA); interH = max(0, yB-yA)
    interA = interW*interH
    areaA = max(0, (a[2]-a[0]))*max(0, (a[3]-a[1]))
    areaB = max(0, (b[2]-b[0]))*max(0, (b[3]-b[1]))
    union = areaA + areaB - interA + 1e-9
    return interA / union

class Track:
    def __init__(self, tid, bbox, depth_val, pixel_h, frame_idx, gray_frame):
        self.id = tid
        self.box = bbox  # [x1,y1,x2,y2]
        self.centroid = np.array([ (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2 ])
        self.init_pixel_h = pixel_h
        self.meters_per_pixel = (ASSUMED_HEIGHT_M / pixel_h) if pixel_h>0 else None
        self.meters_per_depth_unit = (self.meters_per_pixel / depth_val) if (depth_val and self.meters_per_pixel) else None
        self.prev_depth = depth_val
        self.prev_centroid = self.centroid.copy()
        self.total_distance_m = 0.0
        self.last_seen = frame_idx
        self.age = 0

        self.kps = self._init_kps(gray_frame)
        self.kps_prev = self.kps.copy() if self.kps is not None else None

    def _init_kps(self, gray):
        x1,y1,x2,y2 = [int(v) for v in self.box]
        w = max(2, x2-x1); h = max(2, y2-y1)
        roi = gray[max(0,y1):min(gray.shape[0], y2), max(0,x1):min(gray.shape[1], x2)]
        if roi.size == 0:
            return None
        pts = cv2.goodFeaturesToTrack(roi, mask=None, **gftt_params)
        if pts is None:
            return None

        pts[:,0,0] += x1
        pts[:,0,1] += y1
        return pts

    def update(self, bbox, depth_val, pixel_h, frame_idx, gray, next_gray):

        self.box = bbox
        new_centroid = np.array([ (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2 ])

        if pixel_h and pixel_h>0:
            self.meters_per_pixel = ASSUMED_HEIGHT_M / pixel_h

        if depth_val is not None and self.meters_per_pixel is not None:
            new_mpdu = self.meters_per_pixel / (depth_val + 1e-9)
            if self.meters_per_depth_unit is None:
                self.meters_per_depth_unit = new_mpdu
            else:
                self.meters_per_depth_unit = 0.6*self.meters_per_depth_unit + 0.4*new_mpdu

        lateral_m = 0.0
        if self.kps is None or len(self.kps)==0:

            self.kps = self._init_kps(gray)
            self.kps_prev = self.kps.copy() if self.kps is not None else None

        if self.kps is not None and self.kps_prev is not None:

            p0 = self.kps_prev
            p1, st, err = cv2.calcOpticalFlowPyrLK(gray, next_gray, p0, None, **lk_params)
            if p1 is not None and st is not None:
                good0 = p0[st==1]
                good1 = p1[st==1]
                if len(good0) > 0:
                    disp = np.linalg.norm(good1 - good0, axis=1)
                    avg_px = float(np.median(disp))
                    if self.meters_per_pixel:
                        lateral_m = avg_px * self.meters_per_pixel
                    self.kps_prev = good1.reshape(-1,1,2)
                    self.kps = self.kps_prev.copy()
            else:
                self.kps = None
                self.kps_prev = None
        dz_m = 0.0
        if (self.meters_per_depth_unit is not None) and (self.prev_depth is not None) and (depth_val is not None):
            dz_m = (self.prev_depth - depth_val) * self.meters_per_depth_unit

        step = math.sqrt(lateral_m**2 + dz_m**2)
        self.total_distance_m += step

        self.prev_centroid = new_centroid.copy()
        self.prev_depth = depth_val
        self.centroid = new_centroid.copy()
        self.last_seen = frame_idx
        self.age = 0
    def mark_missed(self):
        self.age += 1

device = "cuda" if USE_GPU else "cpu"
print("Device:", device)
yolo = YOLO(YOLO_MODEL)

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

cap = cv2.VideoCapture(VIDEO_PATH if VIDEO_PATH != "0" else 0)
if not cap.isOpened():
    raise RuntimeError("Cannot open video source")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = None
if OUTPUT_PATH:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (W, H))

tracks = []
next_tid = 0
frame_idx = 0
start_time = time.time()

interactive_scale_px = None
interactive_scale_m = None
calibrated = False
use_interactive = False


cal_pts = []
def on_mouse(event, x, y, flags, param):
    global cal_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        cal_pts.append((x,y))
        print("Cal point:", (x,y))

cv2.namedWindow("YOLO+MiDaS+OptFlow")
cv2.setMouseCallback("YOLO+MiDaS+OptFlow", on_mouse)

prev_gray = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    results = yolo.predict(frame, imgsz=640, conf=MIN_CONF, verbose=False)
    r = results[0]
    boxes = []
    if r.boxes is not None and len(r.boxes)>0:
        for box in r.boxes:
            cls = int(box.cls.cpu().numpy()) if hasattr(box, "cls") else int(box.cls)
            if cls != 0:  # person only
                continue
            xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, "cpu") else box.xyxy[0].numpy()
            conf = float(box.conf.cpu().numpy()) if hasattr(box.conf, "cpu") else float(box.conf)
            boxes.append([float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])])

    input_tensor = midas_transforms(frame_rgb).to(device)
    with torch.no_grad():
        pred = midas(input_tensor)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()
    depth_map = pred

    detections = []
    for bb in boxes:
        x1,y1,x2,y2 = bb
        w = max(1, x2-x1); h = max(1, y2-y1)
        cx = x1 + w/2; cy = y1 + h/2
        xi = int(np.clip(cx, 2, W-3)); yi = int(np.clip(cy, 2, H-3))
        window = depth_map[yi-2:yi+3, xi-2:xi+3]
        depth_val = float(np.median(window)) if window.size>0 else None
        detections.append({
            "xyxy": bb,
            "centroid": np.array([cx,cy]),
            "pixel_h": h,
            "depth": depth_val
        })
    assigned = set()
    if len(tracks)>0 and len(detections)>0:
        iou_mat = np.zeros((len(tracks), len(detections)), dtype=float)
        for ti, tr in enumerate(tracks):
            for di, det in enumerate(detections):
                iou_mat[ti, di] = iou(tr.box, det["xyxy"])
        while True:
            t_idx, d_idx = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
            max_iou = iou_mat[t_idx, d_idx]
            if max_iou < IOU_MATCH_THRESH:
                break

            if prev_gray is not None:
                tracks[t_idx].update(detections[d_idx]["xyxy"], detections[d_idx]["depth"], detections[d_idx]["pixel_h"], frame_idx, prev_gray, gray)
            else:

                tracks[t_idx].update(detections[d_idx]["xyxy"], detections[d_idx]["depth"], detections[d_idx]["pixel_h"], frame_idx, gray, gray)
            assigned.add(d_idx)
            iou_mat[t_idx, :] = -1
            iou_mat[:, d_idx] = -1
    for di, det in enumerate(detections):
        if di in assigned:
            continue
        tr = Track(next_tid, det["xyxy"], det["depth"], det["pixel_h"], frame_idx, gray)
        tracks.append(tr)
        next_tid += 1
    for tr in tracks:
        if tr.last_seen != frame_idx:
            tr.mark_missed()
    tracks = [t for t in tracks if t.age <= MAX_MISSED]
    info_text = "Press 'c' to calibrate (click 2 points then enter meters), 'h' to use assumed height"
    cv2.putText(frame, info_text, (10, H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    vis = frame.copy()
    for tr in tracks:
        x1,y1,x2,y2 = [int(v) for v in tr.box]
        color = (int((tr.id*37)%255), int((tr.id*91)%255), int((tr.id*53)%255))
        cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
        lbl = f"ID:{tr.id} {tr.total_distance_m:.2f} m"
        cv2.putText(vis, lbl, (x1, max(18,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        cx,cy = int(tr.centroid[0]), int(tr.centroid[1])
        cv2.circle(vis, (cx,cy), 4, color, -1)


    for i, p in enumerate(cal_pts):
        cv2.circle(vis, p, 6, (0,0,255), -1)
        cv2.putText(vis, f"P{i+1}", (p[0]+6, p[1]+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)


    if len(cal_pts) >= 2 and not calibrated:
        px_dist = math.hypot(cal_pts[0][0]-cal_pts[1][0], cal_pts[0][1]-cal_pts[1][1])
        print(f"Pixel dist between points: {px_dist:.2f}. Enter real-world distance in meters (float): ")
        try:
            val = float(input().strip())
            interactive_scale_px = px_dist
            interactive_scale_m = val
            calibrated = True
            use_interactive = True
            print(f"Calibration set: {interactive_scale_m} m = {interactive_scale_px:.2f} px -> {interactive_scale_m/interactive_scale_px:.6f} m/px")
        except Exception as e:
            print("Invalid input, calibration aborted.", e)
            cal_pts = []


    dmin, dmax = depth_map.min(), depth_map.max()
    dv = (depth_map - dmin) / (dmax - dmin + 1e-9)
    dv = (dv * 255).astype(np.uint8)
    dv = cv2.applyColorMap(dv, cv2.COLORMAP_INFERNO)
    dh = 160; dw = int(dh * W / H)
    small = cv2.resize(dv, (dw, dh))
    vis[5:5+dh, 5:5+dw] = small


    if calibrated and use_interactive:
        m_per_px = interactive_scale_m / (interactive_scale_px + 1e-9)
        cv2.putText(vis, f"Calib: {m_per_px:.6f} m/px", (5, 5+dh+20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    else:
        cv2.putText(vis, f"Assumed height: {ASSUMED_HEIGHT_M} m", (5, 5+dh+20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

    cv2.imshow("YOLO+MiDaS+OptFlow", vis)
    if writer: writer.write(vis)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        # start interactive calibration: clear points, user clicks two points then types meters
        cal_pts = []
        print("Click two points on the image (left mouse). Then type the real distance (meters) in console.")
    if key == ord('h'):
        # force use assumed height (reset interactive)
        calibrated = False
        use_interactive = False
        cal_pts = []
        print("Using assumed height calibration.")


    prev_gray = gray.copy()

end_time = time.time()
print(f"Done. Frames processed: {frame_idx}, Time: {end_time-start_time:.1f}s")

cap.release()
if writer: writer.release()
cv2.destroyAllWindows()
