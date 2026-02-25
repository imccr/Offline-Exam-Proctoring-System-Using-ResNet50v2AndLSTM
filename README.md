<div align="center">

<img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/TensorFlow-2.19.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/YOLOv11-purple?style=for-the-badge&logo=yolo&logoColor=white"/>
<img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>

<br/><br/>

```
 ██████╗ ███████╗██████╗ ███████╗
██╔═══██╗██╔════╝██╔══██╗██╔════╝
██║   ██║█████╗  ██████╔╝███████╗
██║   ██║██╔══╝  ██╔═══╝ ╚════██║
╚██████╔╝███████╗██║     ███████║
 ╚═════╝ ╚══════╝╚═╝     ╚══════╝
```

# Offline Exam Proctoring System
### Using Pose Estimation with Fine-Tuned ResNet50V2 on a Custom Dataset

<br/>

> **Real-time suspicious behavior detection in examination halls using computer vision, deep learning, and skeletal pose analysis.**

<br/>

[![Institute](https://img.shields.io/badge/Institute%20of%20Engineering-Thapathali%20Campus-red?style=flat-square)](https://tcioe.edu.np/)
[![Department](https://img.shields.io/badge/Department-Electronics%20%26%20Computer%20Engineering-blue?style=flat-square)]()
[![Batch](https://img.shields.io/badge/Batch-BCT%202079-green?style=flat-square)]()

</div>

---

## 👥 Team

| Name | Roll No |
|------|---------|
| Ramesh Oli | THA079BCT031 |
| Sagar Gupta | THA079BCT037 |
| Shishir Timilsina | THA079BCT040 |
| Swagat Nepal | THA079BCT048 |

---

## 📌 Overview

Exam cheating is a persistent challenge in large examination halls where human invigilators cannot monitor every student simultaneously. **OEPS** addresses this by providing an AI-powered automated proctoring assistant that:

- Detects and tracks every student in the frame
- Analyzes body pose and skeletal structure per student
- Classifies behavior as **Normal** or **Suspicious** in real time
- Logs all suspicious events with timestamps and confidence scores
- Streams live annotated video to a web dashboard

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎯 **YOLOv11s-Pose Detection** | Detects students and extracts 17 body keypoints per person |
| 🦴 **Skeleton ROI Generation** | Converts keypoints into normalized 224×224 skeletal images |
| 🧠 **Fine-Tuned ResNet50V2** | Classifies behavior from skeletal images with per-frame decisions |
| 🔁 **IoU Tracker** | Assigns stable IDs across frames with duplicate suppression |
| 📊 **Live Web Dashboard** | Real-time annotated video stream with stats, probs, and activity log |
| ⚡ **Label Propagation** | Inference every N frames, labels propagated to all frames for smooth output |
| 🎬 **Save Processed Video** | Full-FPS annotated output video with status bar overlay |
| 📈 **Evaluation Module** | Ground truth CSV-based precision, recall, F1 evaluation |

---

## 🏗️ System Architecture

```
Video Footage
      │
      ▼
OpenCV Frame Extraction
      │
      ▼
YOLOv11s-Pose ──────────────────► DeepSORT / IoU Tracker
      │                                     │
      │ Keypoints                           │ Track ID + BBox
      ▼                                     ▼
Skeleton ROI Generator ◄──────────── Matched Tracks
      │
      │ 224×224 Skeletal Image
      ▼
Fine-Tuned ResNet50V2
      │
      │ Behavior Score (0-1)
      ▼
Decision Module (threshold=0.5)
      │
      ▼
Output Interface (Web Dashboard / Video)
```

---

## 🧠 Model Details

### ResNet50V2
- **Base**: ResNet50V2 pretrained on ImageNet
- **Input**: 224×224 RGB skeletal image
- **Output**: Single sigmoid neuron (0 = Normal, 1 = Suspicious)
- **Fine-tuning**: Frozen base → train head → gradual unfreeze with small LR
- **Training data**: Custom dataset recorded in simulated exam hall

### Skeleton Generation Pipeline
```
Raw Keypoints (x, y, confidence)
         │
         ▼
Filter keypoints by confidence > 0.4
         │
         ▼
Center + Scale normalization
         │
         ▼
Draw 14 bone connections on grayscale canvas
         │
         ▼
Dilate → Resize to 224×224 → RGB → ResNet preprocess
```

### Skeleton Connections (14 bones)
```python
[(0,1),(0,2),(1,3),(2,4),          # head
 (0,5),(0,6),                       # neck
 (5,6),                             # shoulders
 (5,7),(7,9),(6,8),(8,10),          # arms
 (5,11),(6,12),(11,12)]             # torso
```

---

## 🗂️ Project Structure

```
OEPS/
├── main.py                  # FastAPI backend + full inference pipeline
├── index.html               # Web dashboard (Tailwind CSS)
├── requirements.txt         # Python dependencies
├── SETUP.md                 # Installation guide
├── uploads/                 # Temporary uploaded videos (auto-created)
└── resnet50v2_final/        # Trained model (not tracked in git)
```

---

## ⚙️ Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `POSE_CONF_THRESHOLD` | 0.6 | YOLO detection confidence |
| `KPT_CONF_THRESHOLD` | 0.4 | Keypoint visibility threshold |
| `SUSPICIOUS_THRESHOLD` | 0.5 | ResNet sigmoid decision boundary |
| `MAX_LOST_FRAMES` | 60 | Frames before track is deleted |
| `SUPPRESS_IOU_THRESHOLD` | 0.5 | Duplicate detection suppression |
| `PROCESS_EVERY_N` | 10 | Inference every Nth frame |

---

## 🚀 Setup & Run

### Prerequisites
- Python 3.12
- CUDA-capable GPU (RTX 4050 or better recommended)
- CUDA 12.1 + cuDNN

### Install

```bash
git clone https://github.com/imccr/Offline-Exam-Proctoring-System-Using-ResNet50v2AndLSTM.git
cd Offline-Exam-Proctoring-System-Using-ResNet50v2AndLSTM
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Add Model
Download `resnet50v2_final.h5` from [Google Drive](#) and place in project root. Update `main.py`:
```python
RESNET_MODEL_PATH = "resnet50v2_final.h5"
```

### Run
```bash
python main.py
```
Open **http://localhost:8000** in your browser.

---

## 🖥️ Web Interface

| Screen | Description |
|--------|-------------|
| **Upload Screen** | Drag & drop or browse to select exam video |
| **Live Monitor** | Annotated video feed with bounding boxes + labels |
| **Live Stats** | Student count, normal/suspicious breakdown, elapsed time |
| **ResNet Probs** | Per-student confidence scores updated every inference frame |
| **Activity Log** | Timestamped log of all suspicious detections |
| **Stop Button** | Instantly stops stream and returns to upload screen |

---

## 📦 Dependencies

```
tensorflow==2.19.0
keras==3.10.0
ultralytics==8.4.16
opencv-python==4.13.0.92
numpy==2.0.2
fastapi
uvicorn
python-multipart
torch
torchvision
```

---

## 📊 Expected Performance

| Metric | Target |
|--------|--------|
| Accuracy | ≥ 96% |
| Precision | ≥ 0.90 |
| Recall | ≥ 0.95 |
| Inference FPS (T4 GPU) | ~14 FPS |
| Inference FPS (RTX 4050) | ~16-18 FPS |

---

## 🗺️ Roadmap

- [x] YOLOv11 pose detection + IoU tracking
- [x] Skeleton ROI generation
- [x] ResNet50V2 fine-tuning on custom dataset
- [x] FastAPI backend with SSE streaming
- [x] Live web dashboard
- [x] Save processed video
- [ ] LSTM temporal analysis integration
- [ ] Multi-camera support
- [ ] Alert notification system

---

<div align="center">

**Institute of Engineering, Thapathali Campus**<br/>
Department of Electronics and Computer Engineering<br/>
BCT 2079 Final Year Project

</div>
