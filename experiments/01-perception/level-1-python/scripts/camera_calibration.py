"""相机标定实验：从棋盘格图像估计内参与畸变参数。

典型流程:
    python scripts/camera_calibration.py --generate-board
    python scripts/camera_calibration.py --capture --input-dir calibration_images
    python scripts/camera_calibration.py --input-dir calibration_images

输出:
    output/camera_calibration.json
    output/camera_calibration/corners_*.jpg
    output/camera_calibration/undistorted_*.jpg
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output"
DEFAULT_INPUT_DIR = BASE_DIR / "calibration_images"
DEFAULT_OUTPUT = OUT_DIR / "camera_calibration.json"
DEFAULT_PREVIEW_DIR = OUT_DIR / "camera_calibration"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def make_object_points(board_cols: int, board_rows: int, square_size: float) -> np.ndarray:
    """生成棋盘格内角点在棋盘坐标系中的 3D 坐标。"""
    points = np.zeros((board_cols * board_rows, 3), np.float32)
    points[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    points *= float(square_size)
    return points


def generate_checkerboard(
    output_path: Path,
    board_cols: int,
    board_rows: int,
    square_px: int,
    margin_squares: int = 1,
) -> Path:
    """生成可打印或屏幕展示的棋盘格图片。

    OpenCV 的 board_cols / board_rows 表示内角点数量，因此实际方格数
    是 (board_cols + 1) x (board_rows + 1)。
    """
    squares_x = board_cols + 1
    squares_y = board_rows + 1
    width = (squares_x + 2 * margin_squares) * square_px
    height = (squares_y + 2 * margin_squares) * square_px
    board = np.full((height, width), 255, dtype=np.uint8)

    offset = margin_squares * square_px
    for y in range(squares_y):
        for x in range(squares_x):
            if (x + y) % 2 == 0:
                top_left = (offset + x * square_px, offset + y * square_px)
                bottom_right = (
                    offset + (x + 1) * square_px,
                    offset + (y + 1) * square_px,
                )
                cv2.rectangle(board, top_left, bottom_right, 0, thickness=-1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), board):
        raise RuntimeError(f"无法写入棋盘格图片: {output_path}")
    return output_path


def find_images(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        return []

    images = [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return images


def display_path(path: Path) -> str:
    """优先输出相对 BASE_DIR 的路径，外部路径则保留绝对路径。"""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(BASE_DIR))
    except ValueError:
        return str(resolved)


def detect_corners(
    image_path: Path,
    board_size: tuple[int, int],
    preview_dir: Path,
) -> tuple[np.ndarray, tuple[int, int]] | None:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  [SKIP] 无法读取: {image_path.name}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, board_size, flags)

    if not found:
        print(f"  [SKIP] 未检测到棋盘格角点: {image_path.name}")
        return None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    preview_dir.mkdir(parents=True, exist_ok=True)
    preview = image.copy()
    cv2.drawChessboardCorners(preview, board_size, refined, found)
    preview_path = preview_dir / f"corners_{image_path.stem}.jpg"
    cv2.imwrite(str(preview_path), preview)

    print(f"  [OK] 角点检测成功: {image_path.name}")
    return refined, (gray.shape[1], gray.shape[0])


def compute_reprojection_errors(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rvecs: tuple[np.ndarray, ...],
    tvecs: tuple[np.ndarray, ...],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> list[float]:
    errors = []
    for idx, points in enumerate(object_points):
        projected, _ = cv2.projectPoints(
            points,
            rvecs[idx],
            tvecs[idx],
            camera_matrix,
            dist_coeffs,
        )
        error = cv2.norm(image_points[idx], projected, cv2.NORM_L2) / len(projected)
        errors.append(float(error))
    return errors


def save_undistorted_preview(
    image_path: Path,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int],
    preview_dir: Path,
) -> Path | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    new_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        image_size,
        alpha=0,
        newImgSize=image_size,
    )
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_matrix)
    x, y, width, height = roi
    if width > 0 and height > 0:
        undistorted = undistorted[y : y + height, x : x + width]

    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / f"undistorted_{image_path.stem}.jpg"
    cv2.imwrite(str(preview_path), undistorted)
    return preview_path


def calibrate_from_images(
    input_dir: Path,
    output_path: Path,
    preview_dir: Path,
    board_cols: int,
    board_rows: int,
    square_size: float,
    min_images: int,
) -> dict:
    image_paths = find_images(input_dir)
    if not image_paths:
        raise RuntimeError(
            f"未找到标定图片: {input_dir}\n"
            "请先运行 --generate-board 生成棋盘格，再用 --capture 或手机拍摄保存图片。"
        )

    board_size = (board_cols, board_rows)
    object_template = make_object_points(board_cols, board_rows, square_size)
    object_points = []
    image_points = []
    accepted_images = []
    image_size = None

    print(f"→ 读取标定图片目录: {input_dir}")
    print(f"→ 棋盘格内角点: {board_cols} x {board_rows}")

    for image_path in image_paths:
        detection = detect_corners(image_path, board_size, preview_dir)
        if detection is None:
            continue

        corners, detected_size = detection
        if image_size is None:
            image_size = detected_size
        elif detected_size != image_size:
            print(
                f"  [SKIP] 尺寸不一致: {image_path.name} "
                f"({detected_size[0]}x{detected_size[1]})"
            )
            continue

        object_points.append(object_template.copy())
        image_points.append(corners)
        accepted_images.append(image_path)

    if len(accepted_images) < min_images:
        raise RuntimeError(
            f"有效标定图片不足: {len(accepted_images)} / {min_images}\n"
            "建议拍摄 15-25 张图片，覆盖画面中心、边缘、近距离、远距离和倾斜角度。"
        )

    print("\n→ 执行 OpenCV 相机标定 …")
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    reprojection_errors = compute_reprojection_errors(
        object_points,
        image_points,
        rvecs,
        tvecs,
        camera_matrix,
        dist_coeffs,
    )

    undistorted_preview = save_undistorted_preview(
        accepted_images[0],
        camera_matrix,
        dist_coeffs,
        image_size,
        preview_dir,
    )

    result = {
        "board": {
            "inner_corners": [board_cols, board_rows],
            "square_size_m": square_size,
        },
        "image_size": {
            "width": image_size[0],
            "height": image_size[1],
        },
        "num_input_images": len(image_paths),
        "num_valid_images": len(accepted_images),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
        "rms_reprojection_error": float(rms),
        "mean_reprojection_error": float(np.mean(reprojection_errors)),
        "per_image_reprojection_error": {
            path.name: reprojection_errors[idx]
            for idx, path in enumerate(accepted_images)
        },
        "accepted_images": [display_path(path) for path in accepted_images],
        "undistorted_preview": (
            display_path(undistorted_preview)
            if undistorted_preview is not None
            else None
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return result


def capture_images(
    input_dir: Path,
    camera_index: int,
    width: int,
    height: int,
    fps: int,
    max_images: int,
    board_cols: int,
    board_rows: int,
    auto_capture: bool,
    capture_interval: float,
) -> int:
    input_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 index={camera_index}")

    board_size = (board_cols, board_rows)
    saved = 0
    last_capture_at = 0.0

    print("→ 摄像头采集模式")
    print("  Space/s: 保存当前帧")
    print("  q: 退出采集")
    if auto_capture:
        print(f"  自动采集: 检测到棋盘格后每 {capture_interval:.1f}s 保存一张")

    try:
        while saved < max_images:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("摄像头读取失败")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, board_size)

            preview = frame.copy()
            if found:
                cv2.drawChessboardCorners(preview, board_size, corners, found)

            status = f"saved {saved}/{max_images} | found={found}"
            cv2.putText(
                preview,
                status,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if found else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("camera_calibration_capture", preview)

            key = cv2.waitKey(1) & 0xFF
            should_save = key in (ord(" "), ord("s"))
            now = time.time()
            if auto_capture and found and now - last_capture_at >= capture_interval:
                should_save = True

            if should_save:
                saved += 1
                last_capture_at = now
                image_path = input_dir / f"calib_{saved:03d}.jpg"
                cv2.imwrite(str(image_path), frame)
                print(f"  [OK] 保存: {image_path}")

            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="相机标定：估计内参和畸变系数")
    parser.add_argument("--generate-board", action="store_true",
                        help="生成棋盘格图片后退出")
    parser.add_argument("--capture", action="store_true",
                        help="打开摄像头采集标定图片，然后执行标定")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="标定图片目录")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="标定结果 JSON 输出路径")
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR,
                        help="角点和去畸变预览输出目录")
    parser.add_argument("--board-output", type=Path, default=None,
                        help="棋盘格图片输出路径")
    parser.add_argument("--board-cols", type=int, default=9,
                        help="棋盘格内角点列数")
    parser.add_argument("--board-rows", type=int, default=6,
                        help="棋盘格内角点行数")
    parser.add_argument("--square-size", type=float, default=0.025,
                        help="棋盘格单个方格边长，单位米")
    parser.add_argument("--square-px", type=int, default=80,
                        help="生成棋盘格时单个方格的像素宽度")
    parser.add_argument("--min-images", type=int, default=3,
                        help="执行标定所需的最少有效图片数")
    parser.add_argument("--camera-index", type=int, default=0,
                        help="OpenCV 摄像头 index，通常 /dev/video0 对应 0")
    parser.add_argument("--width", type=int, default=640,
                        help="摄像头采集宽度")
    parser.add_argument("--height", type=int, default=480,
                        help="摄像头采集高度")
    parser.add_argument("--fps", type=int, default=30,
                        help="摄像头采集帧率")
    parser.add_argument("--max-images", type=int, default=20,
                        help="采集模式最多保存多少张图片")
    parser.add_argument("--auto-capture", action="store_true",
                        help="检测到棋盘格后自动间隔保存")
    parser.add_argument("--capture-interval", type=float, default=1.0,
                        help="自动采集的最小时间间隔，单位秒")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.board_cols <= 0 or args.board_rows <= 0:
        raise ValueError("--board-cols 和 --board-rows 必须为正整数")
    if args.square_size <= 0:
        raise ValueError("--square-size 必须大于 0")
    if args.square_px <= 0:
        raise ValueError("--square-px 必须大于 0")
    if args.min_images < 1:
        raise ValueError("--min-images 必须至少为 1")


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 2

    print("=" * 60)
    print("  相机标定实验 — Camera Calibration")
    print("=" * 60)

    if args.generate_board:
        board_output = args.board_output
        if board_output is None:
            board_output = OUT_DIR / f"checkerboard_{args.board_cols}x{args.board_rows}.png"
        board_path = generate_checkerboard(
            board_output,
            args.board_cols,
            args.board_rows,
            args.square_px,
        )
        print(f"→ 已生成棋盘格: {board_path}")
        print("  可打印或用屏幕展示。拍摄时需要多角度、多距离覆盖画面。")
        if not args.capture:
            return 0

    try:
        if args.capture:
            saved = capture_images(
                args.input_dir,
                args.camera_index,
                args.width,
                args.height,
                args.fps,
                args.max_images,
                args.board_cols,
                args.board_rows,
                args.auto_capture,
                args.capture_interval,
            )
            print(f"→ 本次采集保存 {saved} 张图片")

        result = calibrate_from_images(
            args.input_dir,
            args.output,
            args.preview_dir,
            args.board_cols,
            args.board_rows,
            args.square_size,
            args.min_images,
        )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 1

    print("\n→ 标定结果")
    print(f"  有效图片: {result['num_valid_images']}")
    print(f"  RMS 重投影误差: {result['rms_reprojection_error']:.4f} px")
    print(f"  平均重投影误差: {result['mean_reprojection_error']:.4f} px")
    print(f"  输出 JSON: {args.output}")
    print(f"  预览目录: {args.preview_dir}")
    print("\nDone. Camera calibration complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
