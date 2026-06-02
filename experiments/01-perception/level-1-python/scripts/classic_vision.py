"""经典 OpenCV 视觉基础实验：光流、特征匹配、背景减除。

典型流程:
    python scripts/classic_vision.py --generate-sample --mode optical-flow
    python scripts/classic_vision.py --mode optical-flow --input-video output/classic_vision/sample_motion.mp4
    python scripts/classic_vision.py --mode feature-matching --image-a a.jpg --image-b b.jpg
    python scripts/classic_vision.py --mode background-subtraction --input-video video.mp4

输出:
    output/classic_vision/optical_flow.jpg
    output/classic_vision/feature_matching.jpg
    output/classic_vision/background_subtraction.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output"
DEFAULT_OUTPUT_DIR = OUT_DIR / "classic_vision"

MODES = ("optical-flow", "feature-matching", "background-subtraction")


def draw_sample_frame(index: int, width: int = 640, height: int = 480) -> np.ndarray:
    frame = np.full((height, width, 3), 245, dtype=np.uint8)

    center = (120 + index * 12, 220)
    cv2.circle(frame, center, 45, (40, 120, 230), thickness=-1)
    cv2.rectangle(
        frame,
        (330, 120 + index * 4),
        (430, 220 + index * 4),
        (40, 180, 90),
        thickness=-1,
    )
    cv2.line(frame, (60, 380), (560, 380), (80, 80, 80), 3)
    cv2.putText(
        frame,
        f"sample motion frame {index:02d}",
        (40, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    return frame


def generate_sample_video(output_dir: Path, frame_count: int = 24) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sample_motion.mp4"
    first = draw_sample_frame(0)
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        12,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建样例视频: {output_path}")

    try:
        for idx in range(frame_count):
            writer.write(draw_sample_frame(idx))
    finally:
        writer.release()
    return output_path


def generate_sample_images(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_a = output_dir / "sample_features_a.jpg"
    image_b = output_dir / "sample_features_b.jpg"
    cv2.imwrite(str(image_a), draw_sample_frame(2))
    cv2.imwrite(str(image_b), draw_sample_frame(10))
    return image_a, image_b


def open_capture(input_video: Path | None, camera_index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    if input_video is not None:
        cap = cv2.VideoCapture(str(input_video))
    else:
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        source = input_video if input_video is not None else f"camera index={camera_index}"
        raise RuntimeError(f"无法打开输入源: {source}")
    return cap


def read_two_frames(
    input_video: Path | None,
    camera_index: int,
    width: int,
    height: int,
    fps: int,
) -> tuple[np.ndarray, np.ndarray]:
    cap = open_capture(input_video, camera_index, width, height, fps)
    try:
        ok_a, frame_a = cap.read()
        ok_b = False
        frame_b = None
        for _ in range(8):
            ok_b, frame_b = cap.read()
            if ok_b:
                break
        if not ok_a or not ok_b or frame_b is None:
            raise RuntimeError("输入源至少需要两帧")
        return frame_a, frame_b
    finally:
        cap.release()


def run_optical_flow(
    input_video: Path | None,
    camera_index: int,
    width: int,
    height: int,
    fps: int,
    output_dir: Path,
) -> Path:
    frame_a, frame_b = read_two_frames(input_video, camera_index, width, height, fps)
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    points_a = cv2.goodFeaturesToTrack(
        gray_a,
        maxCorners=80,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7,
    )
    if points_a is None:
        raise RuntimeError("未找到可跟踪特征点")

    points_b, status, _ = cv2.calcOpticalFlowPyrLK(
        gray_a,
        gray_b,
        points_a,
        None,
        winSize=(21, 21),
        maxLevel=3,
    )
    if points_b is None or status is None:
        raise RuntimeError("光流计算失败")

    output = frame_b.copy()
    good_a = points_a[status.flatten() == 1]
    good_b = points_b[status.flatten() == 1]
    for start, end in zip(good_a, good_b):
        x0, y0 = start.ravel().astype(int)
        x1, y1 = end.ravel().astype(int)
        cv2.arrowedLine(output, (x0, y0), (x1, y1), (0, 0, 255), 2, tipLength=0.3)
        cv2.circle(output, (x1, y1), 3, (0, 255, 0), -1)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "optical_flow.jpg"
    cv2.imwrite(str(output_path), output)
    return output_path


def run_feature_matching(image_a: Path, image_b: Path, output_dir: Path) -> Path:
    frame_a = cv2.imread(str(image_a))
    frame_b = cv2.imread(str(image_b))
    if frame_a is None:
        raise RuntimeError(f"无法读取图片: {image_a}")
    if frame_b is None:
        raise RuntimeError(f"无法读取图片: {image_b}")

    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=500)
    keypoints_a, descriptors_a = orb.detectAndCompute(gray_a, None)
    keypoints_b, descriptors_b = orb.detectAndCompute(gray_b, None)

    if descriptors_a is None or descriptors_b is None:
        raise RuntimeError("未找到足够 ORB 特征")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors_a, descriptors_b)
    matches = sorted(matches, key=lambda item: item.distance)[:40]

    output = cv2.drawMatches(
        frame_a,
        keypoints_a,
        frame_b,
        keypoints_b,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "feature_matching.jpg"
    cv2.imwrite(str(output_path), output)
    return output_path


def run_background_subtraction(
    input_video: Path | None,
    camera_index: int,
    width: int,
    height: int,
    fps: int,
    output_dir: Path,
    frames: int,
) -> Path:
    cap = open_capture(input_video, camera_index, width, height, fps)
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=80,
        varThreshold=24,
        detectShadows=True,
    )

    last_frame = None
    last_mask = None
    try:
        for _ in range(frames):
            ok, frame = cap.read()
            if not ok:
                break
            last_frame = frame
            last_mask = subtractor.apply(frame)
    finally:
        cap.release()

    if last_frame is None or last_mask is None:
        raise RuntimeError("输入源没有可用帧")

    mask_bgr = cv2.cvtColor(last_mask, cv2.COLOR_GRAY2BGR)
    output = np.hstack([last_frame, mask_bgr])
    cv2.putText(
        output,
        "left: frame | right: foreground mask",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "background_subtraction.jpg"
    cv2.imwrite(str(output_path), output)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="经典 OpenCV 视觉基础实验")
    parser.add_argument("--mode", choices=MODES, default="optical-flow",
                        help="实验模式")
    parser.add_argument("--generate-sample", action="store_true",
                        help="生成可离线运行的样例视频和图片")
    parser.add_argument("--input-video", type=Path, default=None,
                        help="视频输入；不提供时使用摄像头")
    parser.add_argument("--image-a", type=Path, default=None,
                        help="feature-matching 模式的第一张图片")
    parser.add_argument("--image-b", type=Path, default=None,
                        help="feature-matching 模式的第二张图片")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="输出目录")
    parser.add_argument("--camera-index", type=int, default=0,
                        help="OpenCV 摄像头 index，通常 /dev/video0 对应 0")
    parser.add_argument("--width", type=int, default=640,
                        help="摄像头采集宽度")
    parser.add_argument("--height", type=int, default=480,
                        help="摄像头采集高度")
    parser.add_argument("--fps", type=int, default=30,
                        help="摄像头采集帧率")
    parser.add_argument("--frames", type=int, default=24,
                        help="背景减除处理的最大帧数")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 60)
    print("  经典 OpenCV 视觉基础实验")
    print("=" * 60)

    try:
        if args.generate_sample:
            video_path = generate_sample_video(args.output_dir)
            image_a, image_b = generate_sample_images(args.output_dir)
            print(f"→ 样例视频: {video_path}")
            print(f"→ 样例图片 A: {image_a}")
            print(f"→ 样例图片 B: {image_b}")
            if args.input_video is None:
                args.input_video = video_path
            if args.image_a is None:
                args.image_a = image_a
            if args.image_b is None:
                args.image_b = image_b

        if args.mode == "optical-flow":
            output_path = run_optical_flow(
                args.input_video,
                args.camera_index,
                args.width,
                args.height,
                args.fps,
                args.output_dir,
            )
        elif args.mode == "feature-matching":
            if args.image_a is None or args.image_b is None:
                raise RuntimeError("feature-matching 模式需要 --image-a 和 --image-b")
            output_path = run_feature_matching(args.image_a, args.image_b, args.output_dir)
        else:
            output_path = run_background_subtraction(
                args.input_video,
                args.camera_index,
                args.width,
                args.height,
                args.fps,
                args.output_dir,
                args.frames,
            )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 1

    print(f"→ 输出预览: {output_path}")
    print("\nDone. Classic OpenCV vision lab complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
