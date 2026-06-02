# Classic OpenCV Vision — Optical Flow, Feature Matching, and Background Subtraction

中文： [classic_vision.md](./classic_vision.md)

> Use this tutorial with `scripts/classic_vision.py`. The goal is not to stack many algorithms, but to understand three basic visual signals through runnable experiments: motion, correspondence, and foreground.

---

## 1. Core Question

Robot vision is not only object recognition. It often needs lower-level questions:

- Which points are moving in the image?
- Which local regions correspond between two images?
- Which video regions are newly appearing foreground?

These map to three modes:

| Mode | What You Observe | Typical Use |
|------|------------------|-------------|
| `optical-flow` | Point motion tracks | Tracking, visual odometry, motion estimation |
| `feature-matching` | Local correspondences across two images | Registration, localization, reconstruction |
| `background-subtraction` | Foreground regions | Monitoring, dynamic object detection, scene change |

---

## 2. Generate Sample Data

Go to the Level 1 directory:

```bash
cd experiments/01-perception/level-1-python
```

Generate an offline sample video and sample images:

```bash
python scripts/classic_vision.py --generate-sample --mode optical-flow
```

Output directory:

```text
output/classic_vision/
```

The sample data verifies the environment and algorithm pipeline. It is not a replacement for real camera behavior.

---

## 3. Optical Flow: Observe Motion

```bash
python scripts/classic_vision.py \
  --mode optical-flow \
  --input-video output/classic_vision/sample_motion.mp4
```

The script uses:

- `cv2.goodFeaturesToTrack` to find trackable corners;
- `cv2.calcOpticalFlowPyrLK` to track them between two frames;
- red arrows to show point motion.

Output:

```text
output/classic_vision/optical_flow.jpg
```

Observe whether arrow directions match object motion, and whether arrow lengths reflect motion magnitude.

---

## 4. Feature Matching: Observe Correspondence

Use the sample images first:

```bash
python scripts/classic_vision.py \
  --generate-sample \
  --mode feature-matching
```

Or provide two images:

```bash
python scripts/classic_vision.py \
  --mode feature-matching \
  --image-a image_a.jpg \
  --image-b image_b.jpg
```

The script uses:

- `cv2.ORB_create` to extract ORB features;
- `cv2.BFMatcher` to match descriptors;
- `cv2.drawMatches` to draw match lines between images.

Output:

```text
output/classic_vision/feature_matching.jpg
```

Good matches should connect the same object or local structure. Bad matches often come from repeated texture, blur, or large viewpoint changes.

---

## 5. Background Subtraction: Observe Foreground

```bash
python scripts/classic_vision.py \
  --mode background-subtraction \
  --input-video output/classic_vision/sample_motion.mp4
```

The script uses:

- `cv2.createBackgroundSubtractorMOG2` to build a background model;
- multiple video frames to update the model;
- a side-by-side output of the last frame and foreground mask.

Output:

```text
output/classic_vision/background_subtraction.jpg
```

In the mask, white regions are foreground. Background subtraction works best for fixed cameras and is less reliable when the robot camera moves quickly.

---

## 6. Webcam Mode

If `--input-video` is not passed, optical flow and background subtraction open the webcam by default:

```bash
python scripts/classic_vision.py --mode optical-flow
python scripts/classic_vision.py --mode background-subtraction
```

If the default camera does not open:

```bash
python scripts/classic_vision.py --mode optical-flow --camera-index 1
```

---

## 7. Relation To Embodied AI

These classic methods do not replace deep learning. They provide interpretable visual intuition:

- Optical flow links image motion to robot motion.
- Feature matching explains why localization, SLAM, and reconstruction need stable correspondences.
- Background subtraction introduces dynamic object detection and scene-change reasoning.

After this lab, the Level 1 vision track extends from camera calibration to observable image motion, local correspondence, and foreground segmentation.
