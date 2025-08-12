# Setup Instructions

## Installation

1. Create a virtual environment using Python 3.10:
```bash
py -3.10 -m venv .venv
(Then activate: .venv/Scripts/activate)
```

2. Install dependencies:
```bash
pip install -r requirements.base.txt
pip install -r requirements.txt
```

## Usage

Run the object tracking script:
```bash
python track.py --source sample.mp4 --show-vid --save-vid --yolo-weights yolov8n.pt
```

### Basic Options:
- `--source` - Input source (0 for webcam, path to video file, or RTSP stream)
- `--show-vid` - Display tracking video results
- `--save-vid` - Save video tracking results
- `--save-txt` - Save tracking results to text files

### Example Commands:
```bash
# Use webcam with YOLOv8n
python track.py --source 0 --show-vid --yolo-weights yolov8n.pt

# Process video file
python track.py --source video.mp4 --save-vid --yolo-weights yolov8n.pt

# RTSP stream
python track.py --source rtsp://camera_ip:port/stream --yolo-weights yolov8n.pt
```