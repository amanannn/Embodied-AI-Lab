# Camera Calibration — Turning a Webcam into a Usable Vision Sensor

中文： [camera_calibration.md](./camera_calibration.md)

> Use this tutorial with `scripts/camera_calibration.py`. It assumes basic Python and OpenCV knowledge. The goal is to turn a normal USB webcam into a calibrated vision sensor that can later support ArUco pose estimation and other robot vision tasks.

---

## 1. Core Question

A normal webcam only gives you 2D images. Robot vision needs more precise answers:

- Which camera ray does a pixel correspond to?
- Why do image edges bend, and how can we compensate for that?
- Where do the camera parameters for ArUco, AprilTag, and PnP pose estimation come from?

Camera calibration estimates two groups of parameters:

| Parameter | Meaning | Use |
|-----------|---------|-----|
| `camera_matrix` | Focal lengths `fx/fy` and principal point `cx/cy` | Projection between camera coordinates and pixels |
| `dist_coeffs` | Radial and tangential distortion coefficients | Correct lens distortion from low-cost or wide-angle lenses |

Without calibration, later 3D pose estimates may look runnable but have weak physical meaning.

---

## 2. Inputs and Outputs

The input is a set of checkerboard images captured from different viewpoints. The default board has `9 x 6` inner corners, which are the black-white intersections OpenCV detects.

Outputs:

| File | Description |
|------|-------------|
| `output/checkerboard_9x6.png` | Generated checkerboard image |
| `calibration_images/*.jpg` | Captured or manually collected calibration images |
| `output/camera_calibration.json` | Intrinsics, distortion coefficients, and error metrics |
| `output/camera_calibration/corners_*.jpg` | Corner detection previews |
| `output/camera_calibration/undistorted_*.jpg` | Undistortion preview |

---

## 3. Environment

Go to the Level 1 perception directory:

```bash
cd experiments/01-perception/level-1-python
pip install -r requirements.txt
```

If you use conda:

```bash
conda activate embodied-ai
python scripts/camera_calibration.py --generate-board
```

Recommended hardware: a normal USB webcam. `640x480 @ 30fps` is enough. This experiment does not require a depth camera.

---

## 4. Generate the Checkerboard

Generate the default `9 x 6` inner-corner checkerboard:

```bash
python scripts/camera_calibration.py --generate-board
```

Default output:

```text
output/checkerboard_9x6.png
```

You can use it in two ways:

- Print it and keep the paper as flat as possible.
- Display it on another screen and point the camera at the screen.

Printing is usually better because screen glare and moire patterns can reduce corner quality.

---

## 5. Capture Images

Capture images with the webcam:

```bash
python scripts/camera_calibration.py --capture --input-dir calibration_images
```

When the window opens:

| Key | Action |
|-----|--------|
| `Space` or `s` | Save the current frame |
| `q` | Quit capture |

Capture recommendations:

- Use 15-25 images. The script default of 3 valid images is only for first-run accessibility.
- Cover the image center, corners, and edges.
- Include close, far, and tilted views.
- Avoid only front-facing shots, which make the intrinsics unstable.

If you already collected images manually, place them under:

```text
experiments/01-perception/level-1-python/calibration_images/
```

Then run offline calibration directly.

---

## 6. Run Calibration

```bash
python scripts/camera_calibration.py --input-dir calibration_images
```

The script will:

1. Read the images.
2. Detect checkerboard corners with `cv2.findChessboardCorners`.
3. Refine corners with `cv2.cornerSubPix`.
4. Estimate camera parameters with `cv2.calibrateCamera`.
5. Save the JSON result, corner previews, and undistortion preview.

Example output:

```text
→ 标定结果
  有效图片: 18
  RMS 重投影误差: 0.3421 px
  平均重投影误差: 0.0478 px
  输出 JSON: output/camera_calibration.json
```

---

## 7. How to Judge Quality

Start with three signals:

| Signal | Good Result | Common Problem |
|--------|-------------|----------------|
| Valid image count | 15-25 images | Too few images, narrow viewpoint range |
| RMS reprojection error | Usually below 1.0 px | Above 2.0 px often means poor image quality |
| Corner previews | Corners align with the board | Wrong corners, blur, glare |

The most important part of `camera_matrix` is:

```json
[
  [fx, 0, cx],
  [0, fy, cy],
  [0, 0, 1]
]
```

`fx/fy` are focal lengths in pixel units. `cx/cy` are the principal point. Later ArUco pose estimation will pass this matrix directly into OpenCV's PnP pipeline.

---

## 8. Troubleshooting

### The checkerboard is not detected

Check that `--board-cols` and `--board-rows` mean inner corners, not squares. The default `9 x 6` inner corners correspond to a physical board with `10 x 7` squares.

### The calibration error is large

This usually comes from narrow viewpoints, blur, a non-flat board, glare, or board positions only near the image center. Recapturing better images is usually more effective than tuning parameters.

### The webcam does not open

On Linux, `/dev/video0` usually maps to `--camera-index 0`. If you have multiple cameras, try:

```bash
python scripts/camera_calibration.py --capture --camera-index 1
```

### The GUI window cannot open

Skip `--capture`, place images manually under `calibration_images/`, then run:

```bash
python scripts/camera_calibration.py --input-dir calibration_images
```

---

## 9. Next Steps

Camera calibration is the entry point of the vision track. After it, continue with:

- ArUco / AprilTag detection and 6DoF pose estimation.
- PnP, projection, and reprojection error in monocular vision.
- Visual servoing and target tracking experiments.

In the embodied AI loop, calibration answers a simple but necessary question: before the robot understands the world, what exactly is its camera?
