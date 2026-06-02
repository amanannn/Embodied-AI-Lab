# ArUco / AprilTag Pose Estimation — Understanding 6DoF From Visual Markers

中文： [aruco_pose.md](./aruco_pose.md)

> Use this tutorial with `scripts/aruco_pose.py`. It assumes you have completed, or at least understand, `camera_calibration.en.md`, because physically meaningful pose estimation needs camera intrinsics and distortion coefficients.

---

## 1. Core Question

Camera calibration answers what the camera is. ArUco / AprilTag pose estimation asks the next questions:

- Is there a known visual marker in the image?
- Where are the four marker corners in pixels?
- If the marker side length is known, what is its 3D pose relative to the camera?

Visual markers are common in robot labs for quick localization, hand-eye calibration, docking, tabletop manipulation, and visual servoing.

---

## 2. Inputs and Outputs

| Item | Description |
|------|-------------|
| Input marker | `output/aruco_pose/marker_DICT_4X4_50_0.png` |
| Input image | A marker photo or a live webcam frame |
| Optional calibration | `output/camera_calibration.json` |
| Preview output | `output/aruco_pose/aruco_detection.jpg` |
| Data output | `output/aruco_pose/aruco_detection.json` |

Without a calibration file, the script only detects marker IDs and corners. With calibration and a real `--marker-size`, it estimates `rvec` and `tvec_m`.

---

## 3. Generate a Marker

```bash
cd experiments/01-perception/level-1-python
python scripts/aruco_pose.py --generate-marker
```

Default output:

```text
output/aruco_pose/marker_DICT_4X4_50_0.png
```

You can print it or show it on another screen. If printed, measure the real side length and pass it through `--marker-size`.

---

## 4. Detect an Offline Image

If you already have a marker photo:

```bash
python scripts/aruco_pose.py --input-image path/to/marker_photo.jpg
```

This detects the marker and saves an annotated preview, but it does not produce reliable 3D pose.

If camera calibration is available:

```bash
python scripts/aruco_pose.py \
  --input-image path/to/marker_photo.jpg \
  --calibration output/camera_calibration.json \
  --marker-size 0.05
```

`--marker-size 0.05` means the marker side length is 5 cm.

---

## 5. Live Webcam Mode

```bash
python scripts/aruco_pose.py \
  --calibration output/camera_calibration.json \
  --marker-size 0.05
```

Place the marker in front of the webcam. Press `q` to quit.

If your webcam is not `/dev/video0`, try:

```bash
python scripts/aruco_pose.py --camera-index 1
```

---

## 6. Code Walkthrough

The script follows this pipeline:

1. `cv2.aruco.getPredefinedDictionary` selects a marker dictionary.
2. `cv2.aruco.ArucoDetector` detects marker corners and IDs.
3. `cv2.aruco.estimatePoseSingleMarkers` estimates pose from corners, camera intrinsics, and real side length.
4. `cv2.drawFrameAxes` draws the camera-coordinate axes on the image.
5. The script saves an annotated image and a JSON result.

The key JSON fields are:

```json
{
  "id": 0,
  "rvec": [0.1, 0.2, 0.3],
  "tvec_m": [0.0, 0.0, 0.5]
}
```

`tvec_m` is the marker origin position in the camera coordinate frame, in meters. The third value is usually the approximate depth from the camera.

---

## 7. Troubleshooting

### The marker ID is detected, but no pose is shown

Check whether `--calibration` is passed, or whether the default `output/camera_calibration.json` exists.

### The pose values are clearly wrong

First check that `--marker-size` is the real side length in meters. Then check calibration quality.

### Detection is unstable

Make the marker large enough in the image, keep edges sharp, and use even lighting. Distance, blur, and glare all make corners unstable.

---

## 8. Next Step

After this lab, you have a basic visual localization chain:

```text
camera calibration -> marker detection -> single-target 6DoF pose estimation
```

This chain can be extended to visual servoing, robot docking, hand-eye calibration, and tabletop manipulation.
