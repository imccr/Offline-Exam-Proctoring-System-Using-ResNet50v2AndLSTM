# 🧠 Online Exam Proctoring System (OEPS)

An AI-based online exam proctoring system that detects suspicious student behavior (head, hand, body movement) using deep learning and computer vision.

This project uses ResNet-based visual models, real-time video processing, and a FastAPI backend for scalable inference.

---

## 🚀 Features

- 🎥 Video-based exam monitoring
- 👀 Detection of suspicious behaviors:
  - Head movement
  - Hand movement
  - Body movement
  - Normal behavior
- 🧠 Deep learning–based inference (CNN / Keras models)
- ⚡ FastAPI backend for real-time processing
- 📊 Designed for extensibility (multi-model, multi-behavior)

---

## 🗂️ Project Structure

```
OEPS/
├── app/                  # FastAPI backend
│   ├── main.py
│   ├── api/
│   ├── core/
│   ├── services/
│   └── main.py
├── models_data/           # Trained model files (NOT tracked in git)
├── notebooks/             # Training & experimentation notebooks
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

---

## ⚠️ Important Notes

- Trained models (`.h5`, `.pt`, `.onnx`) are **NOT included** in the repository.
- Video uploads and outputs are ignored by git.
- You **must manually place trained models** inside `models_data/`.
- Ignoring this and committing models/videos → your repo becomes trash.

---

## 🛠️ Tech Stack

- Python 3.9+
- PyTorch
- OpenCV
- Ultralytics (YOLO)
- FastAPI
- Uvicorn
- MediaPipe
- NumPy / Pillow / Scikit-learn

---

## 📦 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/OEPS.git
cd OEPS
```

### 2️⃣ Create Virtual Environment (Recommended)
```bash
python -m venv venv
# Linux / Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4️⃣ Add Trained Models
Download trained model files and place them here:

```
models_data/
├── behavior_classifier.pt
├── yolo_model.pt
```

Model paths are loaded dynamically by the backend.

### 5️⃣ Run the Backend Server
```bash
uvicorn app.main:app --reload
```

Server URL: [http://127.0.0.1:8000](http://127.0.0.1:8000)  
API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🧪 Model Training (Optional)

Training notebooks are in `notebooks/` and were used in Google Colab / Kaggle for:

- Dataset preparation
- Model training
- Evaluation

> ⚠️ Training is **NOT required** to run inference.

---

## 🔐 Environment Variables (Optional)

Create a `.env` file if required:

```
MODEL_DIR=models_data
```

---

## 🧯 Common Mistakes (DON’T DO THIS)

- ❌ Don’t commit `.h5`, `.pt`, `.onnx` files  
- ❌ Don’t commit uploaded exam videos  
- ❌ Don’t run training inside FastAPI  
- ❌ Don’t mix notebooks with backend logic  

---

## 📌 Future Improvements

- Multi-label behavior detection  
- Temporal modeling (LSTM / Transformer)  
- Cheating score aggregation  
- Dashboard for invigilators  
- Deployment with Docker  

---

