import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import time
import urllib.request
import os

# 1. Download model if not present
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )
    print("Model downloaded.")

# 2. Initialize MediaPipe HandLandmarker (new Tasks API)
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.8
)
detector = mp_vision.HandLandmarker.create_from_options(options)

# Hand connection pairs (MediaPipe hand topology)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

# Helper: draw landmarks on frame using OpenCV
def draw_landmarks_on_image(img, detection_result):
    if not detection_result.hand_landmarks:
        return
    h, w = img.shape[:2]
    for hand_landmarks in detection_result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], (0, 200, 0), 2)
        for pt in pts:
            cv2.circle(img, pt, 4, (0, 0, 255), cv2.FILLED)

# 2. Settings & UI
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

colors = {
    "RED": (45, 45, 240),
    "BLUE": (240, 150, 45),
    "GREEN": (45, 200, 45),
    "BLACK": (20, 20, 20), 
    "SHAPES": (255, 165, 0), # Orange
    "UI_GREY": (200, 200, 200)
}

draw_color = colors["RED"]
shape_mode = False 
brush_thickness = 15
eraser_thickness = 100
xp, yp = 0, 0 
current_stroke_points = [] 
last_detected_shape = "None"

img_canvas = np.zeros((720, 1280, 3), np.uint8)

# --- SHAPE ENGINE ---
def recognize_shape(points):
    global last_detected_shape
    if len(points) < 20: return None
    
    cnt = np.array(points).reshape((-1, 1, 2)).astype(np.int32)
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    x, y, w, h = cv2.boundingRect(cnt)
    
    # 1. Circle Check (Using Circularity Ratio)
    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    circle_area = np.pi * (radius ** 2)
    circularity = area / circle_area if circle_area > 0 else 0
    
    if circularity > 0.6: 
        last_detected_shape = "CIRCLE"
        return ("CIRCLE", (int(cx), int(cy), int(radius)))

    # 2. Star/Heart Check (Using Solidity)
    if solidity < 0.50: 
        last_detected_shape = "STAR"
        return ("STAR", (x + w//2, y + h//2, w//2))
    if 0.50 <= solidity <= 0.75:
        last_detected_shape = "HEART"
        return ("HEART", (x, y, w, h))

    # 3. Polygon Check (Triangle/Square)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.045 * peri, True)
    vertices = len(approx)

    if vertices == 3:
        last_detected_shape = "TRIANGLE"
        return ("TRIANGLE", approx)
    else:
        last_detected_shape = "SQUARE"
        return ("SQUARE", (x, y, w, h))

def draw_star(canvas, center, size, col, thick):
    pts = []
    for i in range(10):
        r = size if i%2==0 else size//2
        a = i * np.pi / 5 - np.pi/2
        pts.append([center[0] + r*np.cos(a), center[1] + r*np.sin(a)])
    cv2.polylines(canvas, [np.array(pts, np.int32)], True, col, thick)

def draw_heart(canvas, x, y, w, h, col, thick):
    pts = np.array([[x+w//2, y+h], [x, y+h//3], [x+w//4, y], [x+w//2, y+h//4], [x+3*w//4, y], [x+w, y+h//3]], np.int32)
    cv2.polylines(canvas, [pts], True, col, thick)

def commit_shape():
    global current_stroke_points
    if shape_mode and current_stroke_points:
        res = recognize_shape(current_stroke_points)
        if res:
            stype, data = res
            s_col = colors["SHAPES"]
            if stype == "TRIANGLE": cv2.drawContours(img_canvas, [data], 0, s_col, brush_thickness)
            elif stype == "STAR": draw_star(img_canvas, (data[0], data[1]), data[2], s_col, brush_thickness)
            elif stype == "HEART": draw_heart(img_canvas, data[0], data[1], data[2], data[3], s_col, brush_thickness)
            elif stype == "SQUARE": cv2.rectangle(img_canvas, (data[0], data[1]), (data[0]+data[2], data[1]+data[3]), s_col, brush_thickness)
            elif stype == "CIRCLE": cv2.circle(img_canvas, (data[0], data[1]), data[2], s_col, brush_thickness)
        current_stroke_points = []

# --- MAIN LOOP ---
while True:
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1) 
    img_temp = np.zeros_like(img)

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    results = detector.detect(mp_img)
    draw_landmarks_on_image(img, results)

    lm_list = []
    if results.hand_landmarks:
        for hand_lms in results.hand_landmarks:
            for id, lm in enumerate(hand_lms):
                lm_list.append([id, int(lm.x * 1280), int(lm.y * 720)])

    if len(lm_list) != 0:
        x1, y1 = lm_list[8][1:]
        x2, y2 = lm_list[12][1:]
        index_up = lm_list[8][2] < lm_list[6][2]
        middle_up = lm_list[12][2] < lm_list[10][2]

        # SELECTION MODE
        if index_up and middle_up:
            commit_shape() 
            xp, yp = 0, 0 
            cv2.rectangle(img, (x1, y1-25), (x2, y2+25), (255, 255, 255), cv2.FILLED)
            
            if y1 < 100:
                if 20 < x1 < 180: draw_color = colors["RED"]; shape_mode = False
                elif 200 < x1 < 360: draw_color = colors["BLUE"]; shape_mode = False
                elif 380 < x1 < 540: draw_color = colors["GREEN"]; shape_mode = False
                elif 560 < x1 < 720: draw_color = colors["BLACK"]; shape_mode = False
                elif 740 < x1 < 900: shape_mode = True; draw_color = colors["SHAPES"]
                elif 920 < x1 < 1080: draw_color = (0, 0, 0); shape_mode = False
                elif 1100 < x1 < 1260: img_canvas[:] = 0

        # DRAWING MODE
        elif index_up and not middle_up:
            if shape_mode:
                current_stroke_points.append((x1, y1))
                for i in range(1, len(current_stroke_points)):
                    cv2.line(img_temp, current_stroke_points[i-1], current_stroke_points[i], colors["SHAPES"], brush_thickness)
            else:
                if xp == 0 and yp == 0: xp, yp = x1, y1
                thickness = eraser_thickness if draw_color == (0,0,0) else brush_thickness
                cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, thickness)
                xp, yp = x1, y1
        else:
            commit_shape()
            xp, yp = 0, 0

    # Layer Blending
    combined_dwg = cv2.addWeighted(img_canvas, 1, img_temp, 1, 0)
    img_gray = cv2.cvtColor(combined_dwg, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 5, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, combined_dwg)

    # UI HEADER
    cv2.rectangle(img, (0, 0), (1280, 100), (40, 40, 40), cv2.FILLED)
    btns = [("RED", 20, colors["RED"]), ("BLUE", 200, colors["BLUE"]), ("GREEN", 380, colors["GREEN"]), 
            ("BLACK", 560, colors["BLACK"]), ("SHAPES", 740, colors["SHAPES"]), ("ERASER", 920, colors["UI_GREY"]), ("CLEAR", 1100, colors["UI_GREY"])]
    for n, x, c in btns:
        is_active = (not shape_mode and draw_color == c and n not in ["ERASER", "SHAPES"]) or \
                    (not shape_mode and draw_color == (0,0,0) and n == "ERASER") or \
                    (shape_mode and n == "SHAPES")
        if is_active:
            cv2.rectangle(img, (x-2, 18), (x+162, 82), (255, 255, 255), 3)
        cv2.rectangle(img, (x, 20), (x+160, 80), c, cv2.FILLED)
        cv2.putText(img, n, (x+25, 58), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    # Status Overlay
    mode_text = "SHAPE MODE" if shape_mode else "WRITING MODE"
    cv2.putText(img, f"MODE: {mode_text}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    if shape_mode:
        cv2.putText(img, f"LAST SHAPE: {last_detected_shape}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

    cv2.imshow("AI Painter Pro - Final Build", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()