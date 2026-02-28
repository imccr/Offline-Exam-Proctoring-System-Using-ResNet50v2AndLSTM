# OEPS v2.0 — Local Setup (Windows + RTX 4050)

## 1. Install Python 3.10
Download from https://www.python.org/downloads/release/python-3100/
Check "Add to PATH" during install.

## 2. Install CUDA + cuDNN
- CUDA 12.1: https://developer.nvidia.com/cuda-12-1-0-download-archive
- cuDNN: https://developer.nvidia.com/cudnn (requires free NVIDIA account)

## 3. Create virtual environment
Open CMD in the project folder:
  python -m venv venv
  venv\Scripts\activate

## 4. Install dependencies
pip install -r requirements.txt

## 5. Install TailwindCSS
https://github.com/tailwindlabs/tailwindcss/releases/latest/download/tailwindcss-windows-x64.exe

## 6. Download model
- Download resnet50v2_final.keras from Google Drive
- Place it in the project folder
- Update RESNET_MODEL_PATH in main.py

## 6. Run Backend
  python main.py

## 7. Run CSS
./tailwindcss -i input.css -o output.css --watch

## 8. Open browser
  http://localhost:8000
