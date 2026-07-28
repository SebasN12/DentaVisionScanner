## Requirements

- Python 3.11–3.12 recommended

## Install dependencies

```bash
pip install -r requirements.txt
```

## Structure

```bash
openCV/
│
├── data/
├── output/
├── main.py
│
└── src/
    │
    ├── __init__.py
    │
    ├── core/
    │   ├── __init__.py
    │   ├── frame.py
    │   ├── match_result.py
    │   └── reconstruction.py
    │
    ├── pipeline/
    │   ├── __init__.py
    │   ├── camera.py
    │   ├── features.py
    │   ├── matching.py
    │   ├── pose.py
    │   ├── triangulation.py
    │   └── pointcloud.py
    │
    ├── visualization/
    │   ├── __init__.py
    │   └── visualizer.py
    │
    └── utils/
        └── __init__.py
```