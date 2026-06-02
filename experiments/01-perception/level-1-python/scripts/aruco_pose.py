"""ArUco / AprilTag 检测与单标记位姿估计实验。

典型流程:
    python scripts/aruco_pose.py --generate-marker
    python scripts/aruco_pose.py --input-image path/to/marker_photo.jpg
    python scripts/aruco_pose.py --input-image path/to/marker_photo.jpg --calibration output/camera_calibration.json

输出:
    output/aruco_pose/marker_DICT_4X4_50_0.png
    output/aruco_pose/aruco_detection.jpg
    output/aruco_pose/aruco_detection.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output"
DEFAULT_OUTPUT_DIR = OUT_DIR / "aruco_pose"
DEFAULT_CALIBRATION = OUT_DIR / "camera_calibration.json"


ARUCO_DICTIONARIES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


def get_dictionary(name: str) -> cv2.aruco.Dictionary:
    if name not in ARUCO_DICTIONARIES:
        choices = ", ".join(sorted(ARUCO_DICTIONARIES))
        raise ValueError(f"未知 ArUco 字典: {name}. 可选: {choices}")
    return cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[name])


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    return camera_matrix, dist_coeffs


def generate_marker(
    dictionary_name: str,
    marker_id: int,
    side_pixels: int,
    output_dir: Path,
) -> Path:
    dictionary = get_dictionary(dictionary_name)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, side_pixels)
    margin = max(20, side_pixels // 8)
    canvas = np.full(
        (side_pixels + 2 * margin, side_pixels + 2 * margin),
        255,
        dtype=np.uint8,
    )
    canvas[margin : margin + side_pixels, margin : margin + side_pixels] = marker

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"marker_{dictionary_name}_{marker_id}.png"
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"无法写入 marker 图片: {output_path}")
    return output_path


def create_detector(dictionary_name: str) -> cv2.aruco.ArucoDetector:
    dictionary = get_dictionary(dictionary_name)
    parameters = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def estimate_single_marker_pose(
    corners: tuple[np.ndarray, ...],
    marker_size: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """用 OpenCV 的单标记 PnP 封装估计 rvec / tvec。"""
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners,
        marker_size,
        camera_matrix,
        dist_coeffs,
    )
    return rvecs, tvecs


def annotate_frame(
    frame: np.ndarray,
    corners: tuple[np.ndarray, ...],
    ids: np.ndarray | None,
    rvecs: np.ndarray | None,
    tvecs: np.ndarray | None,
    camera_matrix: np.ndarray | None,
    dist_coeffs: np.ndarray | None,
    axis_length: float,
) -> np.ndarray:
    annotated = frame.copy()

    if ids is None or len(ids) == 0:
        cv2.putText(
            annotated,
            "No marker detected",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

    if (
        rvecs is not None
        and tvecs is not None
        and camera_matrix is not None
        and dist_coeffs is not None
    ):
        for idx, marker_id in enumerate(ids.flatten()):
            cv2.drawFrameAxes(
                annotated,
                camera_matrix,
                dist_coeffs,
                rvecs[idx],
                tvecs[idx],
                axis_length,
            )
            translation = tvecs[idx].reshape(-1)
            label = (
                f"id={int(marker_id)} "
                f"t=({translation[0]:.2f},{translation[1]:.2f},{translation[2]:.2f})m"
            )
            top_left = corners[idx][0][0].astype(int)
            cv2.putText(
                annotated,
                label,
                (int(top_left[0]), max(20, int(top_left[1]) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    return annotated


def detection_result(
    ids: np.ndarray | None,
    rvecs: np.ndarray | None,
    tvecs: np.ndarray | None,
    dictionary_name: str,
    marker_size: float,
    used_calibration: bool,
) -> dict:
    marker_ids = [] if ids is None else [int(value) for value in ids.flatten()]
    markers = []
    for idx, marker_id in enumerate(marker_ids):
        item = {"id": marker_id}
        if rvecs is not None and tvecs is not None:
            item["rvec"] = rvecs[idx].reshape(-1).astype(float).tolist()
            item["tvec_m"] = tvecs[idx].reshape(-1).astype(float).tolist()
        markers.append(item)

    return {
        "dictionary": dictionary_name,
        "marker_size_m": marker_size,
        "used_calibration": used_calibration,
        "num_markers": len(marker_ids),
        "markers": markers,
    }


def detect_in_frame(
    frame: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    dictionary_name: str,
    marker_size: float,
    camera_matrix: np.ndarray | None,
    dist_coeffs: np.ndarray | None,
    axis_length: float,
) -> tuple[np.ndarray, dict]:
    corners, ids, _ = detector.detectMarkers(frame)

    rvecs = None
    tvecs = None
    if ids is not None and camera_matrix is not None and dist_coeffs is not None:
        rvecs, tvecs = estimate_single_marker_pose(
            corners,
            marker_size,
            camera_matrix,
            dist_coeffs,
        )

    annotated = annotate_frame(
        frame,
        corners,
        ids,
        rvecs,
        tvecs,
        camera_matrix,
        dist_coeffs,
        axis_length,
    )
    result = detection_result(
        ids,
        rvecs,
        tvecs,
        dictionary_name,
        marker_size,
        used_calibration=camera_matrix is not None,
    )
    return annotated, result


def detect_image(
    input_image: Path,
    output_dir: Path,
    detector: cv2.aruco.ArucoDetector,
    dictionary_name: str,
    marker_size: float,
    camera_matrix: np.ndarray | None,
    dist_coeffs: np.ndarray | None,
    axis_length: float,
) -> dict:
    frame = cv2.imread(str(input_image))
    if frame is None:
        raise RuntimeError(f"无法读取图片: {input_image}")

    annotated, result = detect_in_frame(
        frame,
        detector,
        dictionary_name,
        marker_size,
        camera_matrix,
        dist_coeffs,
        axis_length,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    image_output = output_dir / "aruco_detection.jpg"
    json_output = output_dir / "aruco_detection.json"
    cv2.imwrite(str(image_output), annotated)
    json_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result["annotated_image"] = str(image_output.relative_to(BASE_DIR))
    result["json_output"] = str(json_output.relative_to(BASE_DIR))
    return result


def run_camera(
    camera_index: int,
    width: int,
    height: int,
    fps: int,
    detector: cv2.aruco.ArucoDetector,
    dictionary_name: str,
    marker_size: float,
    camera_matrix: np.ndarray | None,
    dist_coeffs: np.ndarray | None,
    axis_length: float,
) -> None:
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 index={camera_index}")

    print("→ 摄像头检测模式")
    print("  q: 退出")
    print("  需要把打印或屏幕显示的 marker 放到摄像头前")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("摄像头读取失败")

            annotated, _ = detect_in_frame(
                frame,
                detector,
                dictionary_name,
                marker_size,
                camera_matrix,
                dist_coeffs,
                axis_length,
            )
            cv2.imshow("aruco_pose", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ArUco / AprilTag 检测与位姿估计")
    parser.add_argument("--generate-marker", action="store_true",
                        help="生成 ArUco marker 图片后退出")
    parser.add_argument("--dictionary", default="DICT_4X4_50",
                        choices=sorted(ARUCO_DICTIONARIES),
                        help="ArUco / AprilTag 字典")
    parser.add_argument("--marker-id", type=int, default=0,
                        help="生成或重点观察的 marker id")
    parser.add_argument("--marker-pixels", type=int, default=600,
                        help="生成 marker 图片的边长像素")
    parser.add_argument("--marker-size", type=float, default=0.05,
                        help="真实 marker 边长，单位米，用于 6DoF 位姿估计")
    parser.add_argument("--axis-length", type=float, default=0.03,
                        help="绘制坐标轴长度，单位米")
    parser.add_argument("--input-image", type=Path, default=None,
                        help="离线检测图片路径")
    parser.add_argument("--calibration", type=Path, default=None,
                        help="相机标定 JSON；默认不估计位姿，只检测 marker")
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
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.marker_id < 0:
        raise ValueError("--marker-id 必须为非负整数")
    if args.marker_pixels <= 0:
        raise ValueError("--marker-pixels 必须大于 0")
    if args.marker_size <= 0:
        raise ValueError("--marker-size 必须大于 0")
    if args.axis_length <= 0:
        raise ValueError("--axis-length 必须大于 0")


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 2

    print("=" * 60)
    print("  ArUco / AprilTag 位姿估计实验")
    print("=" * 60)

    try:
        if args.generate_marker:
            marker_path = generate_marker(
                args.dictionary,
                args.marker_id,
                args.marker_pixels,
                args.output_dir,
            )
            print(f"→ 已生成 marker: {marker_path}")
            if args.input_image is None:
                return 0

        camera_matrix = None
        dist_coeffs = None
        calibration_path = args.calibration
        if calibration_path is None and DEFAULT_CALIBRATION.exists():
            calibration_path = DEFAULT_CALIBRATION
        if calibration_path is not None:
            camera_matrix, dist_coeffs = load_calibration(calibration_path)
            print(f"→ 已加载标定文件: {calibration_path}")
        else:
            print("→ 未提供标定文件：只检测 marker，不输出真实 6DoF 位姿")

        detector = create_detector(args.dictionary)
        if args.input_image is not None:
            result = detect_image(
                args.input_image,
                args.output_dir,
                detector,
                args.dictionary,
                args.marker_size,
                camera_matrix,
                dist_coeffs,
                args.axis_length,
            )
            print(f"→ 检测到 marker 数量: {result['num_markers']}")
            print(f"→ 输出图片: {result['annotated_image']}")
            print(f"→ 输出 JSON: {result['json_output']}")
        else:
            run_camera(
                args.camera_index,
                args.width,
                args.height,
                args.fps,
                detector,
                args.dictionary,
                args.marker_size,
                camera_matrix,
                dist_coeffs,
                args.axis_length,
            )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 1

    print("\nDone. ArUco pose lab complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
