# 🎨 AI Virtual Painter

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-00BCD4?style=for-the-badge&logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Draw in the air using just your hand — no touch required.**  
A real-time, gesture-driven painting application powered by computer vision and AI hand tracking.

</div>

---

## ✨ Features

- 🖐️ **Real-Time Hand Tracking** — Uses MediaPipe's Hand Landmarker to detect and track your hand at high accuracy
- 🎨 **Multiple Paint Colors** — Choose from Red, Blue, Green, and Black with a gesture tap on the toolbar
- 🔷 **Smart Shape Recognition** — Draw rough shapes and the AI snaps them to: **Circle, Triangle, Square, Star, Heart**
- 🧹 **Eraser Tool** — Wipe away mistakes with a wide, gesture-controlled eraser
- 🗑️ **Clear Canvas** — One gesture to wipe the entire canvas clean
- 👆 **Two-Finger Selection Mode** — Raise index + middle finger to switch tools without drawing
- ☝️ **One-Finger Drawing Mode** — Lower your middle finger to start painting

---

## 📸 Demo

| Mode | Gesture | Action |
|---|---|---|
| **Selection** | Index + Middle finger up | Navigate toolbar buttons |
| **Drawing** | Index finger up only | Paint / erase on canvas |
| **Shape Mode** | Select `SHAPES` button | Converts strokes to geometric shapes |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.8+** | Core language |
| **OpenCV** | Video capture, image processing & rendering |
| **MediaPipe Tasks API** | AI hand landmark detection |
| **NumPy** | Contour math & shape recognition |

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/MonaliKaushal/ai-virtual-painter.git
cd ai-virtual-painter
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install opencv-python mediapipe numpy
```

### 4. Run the App

```bash
python painter.py
```

> ⚡ The `hand_landmarker.task` model file is included in the repo. If it's missing, the script will automatically download it on first run.

---

## 🎮 How to Use

1. **Launch** the application — your webcam feed will appear with a toolbar at the top.
2. **Raise both index and middle fingers** to enter **Selection Mode** and hover over toolbar buttons.
3. **Lower your middle finger** (index finger only) to enter **Drawing Mode** and start painting.
4. **Select SHAPES** from the toolbar, then draw any rough shape — the app will recognize and render it as a clean geometric shape.
5. Press **`Q`** to quit.

---

## 📁 Project Structure

```
ai-virtual-painter/
│
├── painter.py              # Main application script
├── hand_landmarker.task    # Pre-trained MediaPipe hand landmark model
├── .gitignore              # Files excluded from version control
└── README.md               # Project documentation
```

---

## ⚙️ Configuration

You can tweak these variables at the top of `painter.py`:

| Variable | Default | Description |
|---|---|---|
| `brush_thickness` | `15` | Drawing brush size |
| `eraser_thickness` | `100` | Eraser brush size |
| `num_hands` | `1` | Number of hands to track |
| `min_hand_detection_confidence` | `0.5` | Detection confidence threshold |
| `min_tracking_confidence` | `0.8` | Tracking confidence threshold |

---

## 🧠 How Shape Recognition Works

When **Shape Mode** is active, strokes are collected as a list of points. On releasing the gesture, the app runs a pipeline:

1. **Circularity Ratio** → Detects circles
2. **Solidity Ratio** (hull vs. contour area) → Distinguishes stars and hearts
3. **Polygon Approximation** (`approxPolyDP`) → Identifies triangles vs. quadrilaterals

The detected shape is then rendered cleanly on the canvas.

---

## 🙋‍♀️ Author

**Monali**  
Built with ❤️ using Python, OpenCV, and MediaPipe.

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and share!
