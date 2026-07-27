## Requirements

- Python 3.11–3.12 recommended

## Install dependencies

```bash
pip install -r requirements.txt
```

## Structure

```bash
├── data/
│   ├── calibration/
│   ├── images/
│   └── test_dataset/
│
├── src/
│   ├── camera.py
│   ├── features.py
│   ├── frame.py
│   ├── matching.py
│   ├── pose.py
│   ├── triangulation.py
│   ├── pointcloud.py
│   └── visualization.py
│
├── output/
│   ├── debug/
│   ├── meshes/
│   ├── pointclouds/
│   ├── poses/
│   └── screenshots/
│
└── main.py
```