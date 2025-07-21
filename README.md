# 🚗 **Road Lane Detection** 🚗

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

Short and sweet real‑time lane detection in Python using OpenCV. This project processes video files (5–50 clips) to identify and annotate white and yellow lane markings, handling both solid and dashed lines as well as curves.

## Features

* 📹 **Video Input**: Frame extraction and preprocessing
* 🎨 **Thresholding**: Color (HLS) and gradient (Sobel) filters
* 🔄 **Perspective Transform**: Convert to bird’s‑eye view for robust detection
* 🛣️ **Lane Identification**: Sliding‑window and polynomial fitting
* 📐 **Metrics Calculation**: Radius of curvature and lane position offset

## Quick Start

1. **Clone the repo**

   ```bash
   git clone [https://github.com/yourusername/road-lane-detection.git](https://github.com/zimbakovtech/RoadLaneDetection.git)
   cd RoadLaneDetection
   ```
2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
3. **Run detection**

   ```bash
   python run_lane_detection.py --input data/processed/sample.mp4 --output output/demo.mp4
   ```

## Repository Layout

```
lane_detection/
├── data/           # raw & processed videos
├── src/            # core modules (calibrate, threshold, transform, detect)
├── notebooks/      # demos & experiments
├── tests/          # unit & integration tests
├── requirements.txt
├── run_lane_detection.py
└── LICENSE         # MIT License
```

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

---

*Prepared by Damjan Zimbakov & Efimija Cuneva*  
*July 2025*  
