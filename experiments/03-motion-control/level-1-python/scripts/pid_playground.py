"""Run the c01-pid-control-playground lab.

Usage:
    python scripts/pid_playground.py --serve
    python scripts/pid_playground.py --simulate --kp 8 --ki 1.2 --kd 2.6
"""

from __future__ import annotations

import argparse
import json
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output/pid_playground"
sys.path.insert(0, str(BASE_DIR))

from controllers.pid import simulate_pid  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="c01-pid-control-playground: interactive PID control lab",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--serve",
        action="store_true",
        help="start a local browser playground backed by the Python simulator",
    )
    mode.add_argument(
        "--simulate",
        action="store_true",
        help="run one simulation and write JSON outputs without opening a browser",
    )
    parser.add_argument("--host", default="127.0.0.1", help="server host for --serve")
    parser.add_argument("--port", type=int, default=8003, help="server port for --serve")
    parser.add_argument("--kp", type=float, default=8.0, help="proportional gain")
    parser.add_argument("--ki", type=float, default=1.2, help="integral gain")
    parser.add_argument("--kd", type=float, default=2.6, help="derivative gain")
    parser.add_argument("--setpoint", type=float, default=1.0, help="target position")
    parser.add_argument("--duration", type=float, default=6.0, help="simulation seconds")
    parser.add_argument("--dt", type=float, default=0.02, help="simulation time step")
    parser.add_argument(
        "--disturbance-time",
        type=float,
        default=3.0,
        help="time in seconds when the disturbance starts",
    )
    parser.add_argument(
        "--disturbance-force",
        type=float,
        default=-1.5,
        help="short external force injected after disturbance-time",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for JSON simulation outputs",
    )
    return parser.parse_args()


def simulation_payload(params: dict[str, float]) -> dict[str, object]:
    result = simulate_pid(**params)
    return {"samples": result.samples, "metrics": result.metrics}


def run_simulation(args: argparse.Namespace) -> None:
    params = {
        "setpoint": args.setpoint,
        "kp": args.kp,
        "ki": args.ki,
        "kd": args.kd,
        "duration": args.duration,
        "dt": args.dt,
        "disturbance_time": args.disturbance_time,
        "disturbance_force": args.disturbance_force,
    }
    payload = simulation_payload(params)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pid_playground_metrics.json").write_text(
        json.dumps(payload["metrics"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "pid_playground_samples.json").write_text(
        json.dumps(payload["samples"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    metrics = payload["metrics"]
    print("c01-pid-control-playground")
    print(
        "steady_state_error={steady_state_error:.4f}, "
        "overshoot_percent={overshoot_percent:.2f}, "
        "settling_time={settling_time:.2f}s".format(**metrics)
    )
    print(f"outputs={args.output_dir}")


def parse_float(query: dict[str, list[str]], name: str, default: float) -> float:
    raw = query.get(name, [str(default)])[0]
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


class PlaygroundHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self) -> None:  # noqa: N802 - stdlib hook name
        parsed = urlparse(self.path)
        if parsed.path == "/api/simulate":
            self._handle_simulation(parsed.query)
            return
        if parsed.path in ("/", "/index.html"):
            self._handle_index()
            return
        super().do_GET()

    def _handle_index(self) -> None:
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        payload = simulation_payload(
            {
                "setpoint": 1.0,
                "kp": 8.0,
                "ki": 1.2,
                "kd": 2.6,
                "duration": 6.0,
                "dt": 0.02,
                "disturbance_time": 3.0,
                "disturbance_force": -1.5,
            }
        )
        html = html.replace(
            "__PID_INITIAL_PAYLOAD__",
            json.dumps(payload, ensure_ascii=False),
        )
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_simulation(self, raw_query: str) -> None:
        try:
            query = parse_qs(raw_query)
            params = {
                "setpoint": parse_float(query, "setpoint", 1.0),
                "kp": parse_float(query, "kp", 8.0),
                "ki": parse_float(query, "ki", 1.2),
                "kd": parse_float(query, "kd", 2.6),
                "duration": parse_float(query, "duration", 6.0),
                "dt": 0.02,
                "disturbance_time": parse_float(query, "disturbance_time", 3.0),
                "disturbance_force": parse_float(query, "disturbance_force", -1.5),
            }
            payload = simulation_payload(params)
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except ValueError as exc:
            body = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.BAD_REQUEST)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def serve(host: str, port: int) -> None:
    if not WEB_DIR.is_dir():
        raise FileNotFoundError(f"missing web directory: {WEB_DIR}")
    server = ThreadingHTTPServer((host, port), PlaygroundHandler)
    print("c01-pid-control-playground")
    print(f"open http://{host}:{port}")
    print("press Ctrl+C to stop")
    server.serve_forever()


def main() -> None:
    args = parse_args()
    if args.serve:
        serve(args.host, args.port)
    else:
        run_simulation(args)


if __name__ == "__main__":
    main()
