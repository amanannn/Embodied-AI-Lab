"""
【Web 推理服务模块 - server.py】

【作用】
    基于 Flask 的 Web 推理服务，提供:
    1.  GET  /         → 返回单页 Web 前端（纯 HTML + 内联 JS）
    2.  POST /generate → 接收数字标签，生成手写数字图片，返回 base64

【关键函数】
    index()      - 渲染主页面（HTML）
    api_generate() - REST API: 接收 JSON → 生成图片 → 返回 base64

【输入输出】
    POST /generate
      输入:  {"digit": 5}
      输出:  {"success": true, "digit": 5, "image_base64": "iVBOR...", "method": "rk4", "steps": 50}

    页面可直接选择数字 0~9 + 采样方法，即时生成预览。

【启动命令】
    python server.py --checkpoint checkpoints/ckpt_epoch_0100.pt --port 5000
"""

import os
import io
import base64
import argparse
import torch
from flask import Flask, request, jsonify, send_file

from models import create_model
from generate import sample_euler, sample_rk4

# ---- Flask 应用 ----
app = Flask(__name__)

# 全局模型实例（启动时加载）
MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_TYPE = "unet"


def load_model(checkpoint_path: str):
    """加载训练好的模型到全局变量。"""
    global MODEL, MODEL_TYPE, DEVICE
    print(f"📂 加载模型: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    MODEL_TYPE = ckpt.get("model_type", "unet")
    MODEL = create_model(MODEL_TYPE).to(DEVICE)
    MODEL.load_state_dict(ckpt["model_state_dict"])
    MODEL.eval()
    n_params = sum(p.numel() for p in MODEL.parameters())
    print(f"   ✅ 模型已就绪 | 类型: {MODEL_TYPE.upper()} | "
          f"参数量: {n_params:,} | 设备: {DEVICE}")


@torch.no_grad()
def generate_single(digit: int, method: str = "rk4", num_steps: int = 50) -> str:
    """
    生成单个数字的图像，返回 base64 编码字符串。

    Args:
        digit:     目标数字 0~9
        method:    采样方法 "euler" 或 "rk4"
        num_steps: 积分步数

    Returns:
        base64 PNG 字符串
    """
    label = torch.tensor([digit], dtype=torch.long, device=DEVICE)
    sample_fn = sample_rk4 if method.lower() == "rk4" else sample_euler
    image = sample_fn(MODEL, label, num_steps=num_steps, device=DEVICE)

    # 后处理: [-1, 1] → [0, 1] → [0, 255]
    image = image.clamp(-1, 1)
    image = (image + 1) / 2
    image = (image * 255).to(torch.uint8)

    # 转为 PNG → base64
    from torchvision.transforms.functional import to_pil_image
    pil_img = to_pil_image(image.squeeze(0))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ============================================================
# 路由: 主页面
# ============================================================
@app.route("/")
def index():
    """返回 Web 前端页面。"""
    return HTML_PAGE


# ============================================================
# 路由: 生成 API
# ============================================================
@app.route("/generate", methods=["POST"])
def api_generate():
    """
    POST /generate
    Body: {"digit": 5, "method": "rk4", "steps": 50}
    Response: {"success": true, "digit": 5, "image_base64": "...", "method": "rk4", "steps": 50}
    """
    try:
        data = request.get_json()
        digit = int(data.get("digit", 5))
        method = data.get("method", "rk4")
        steps = int(data.get("steps", 50))

        # 参数校验
        if digit < 0 or digit > 9:
            return jsonify({"success": False, "error": "digit 必须在 0~9 之间"}), 400
        if method not in ("euler", "rk4"):
            return jsonify({"success": False, "error": "method 必须是 'euler' 或 'rk4'"}), 400
        if steps < 1 or steps > 500:
            return jsonify({"success": False, "error": "steps 必须在 1~500 之间"}), 400

        # 生成
        img_b64 = generate_single(digit, method=method, num_steps=steps)

        return jsonify({
            "success": True,
            "digit": digit,
            "image_base64": img_b64,
            "method": method,
            "steps": steps,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# 内联 HTML 前端（单文件，无外部依赖）
# ============================================================
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Flow Matching 条件生成手写数字</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh; display: flex; justify-content: center; align-items: center;
    color: #e0e0e0;
  }
  .container {
    background: rgba(255,255,255,0.05); backdrop-filter: blur(20px);
    border-radius: 24px; padding: 48px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
    max-width: 580px; width: 90%; text-align: center;
  }
  h1 {
    font-size: 1.8rem; font-weight: 700; margin-bottom: 8px;
    background: linear-gradient(90deg, #f093fb, #f5576c, #fda085);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .subtitle { color: #888; font-size: 0.9rem; margin-bottom: 32px; }
  .digit-selector { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; margin-bottom: 20px; }
  .digit-btn {
    width: 44px; height: 44px; border-radius: 12px; border: 2px solid rgba(255,255,255,0.15);
    background: rgba(255,255,255,0.05); color: #ccc; font-size: 1.1rem;
    font-weight: 700; cursor: pointer; transition: all 0.2s;
  }
  .digit-btn:hover { border-color: #f5576c; color: #f5576c; transform: translateY(-2px); }
  .digit-btn.active { background: linear-gradient(135deg, #f093fb, #f5576c); border-color: transparent; color: #fff; }

  .controls { display: flex; gap: 12px; margin-bottom: 24px; justify-content: center; flex-wrap: wrap; }
  select, .generate-btn {
    padding: 10px 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.15);
    background: rgba(255,255,255,0.08); color: #e0e0e0; font-size: 0.95rem; cursor: pointer;
  }
  select:hover { border-color: rgba(255,255,255,0.3); }
  .generate-btn {
    background: linear-gradient(135deg, #f093fb, #f5576c);
    border: none; color: #fff; font-weight: 600; transition: all 0.2s;
  }
  .generate-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(245,87,108,0.3); }
  .generate-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

  .result-area { margin-top: 28px; }
  .result-area img {
    max-width: 200px; border-radius: 12px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    image-rendering: pixelated; /* 保持像素风格 */
  }
  .placeholder {
    width: 200px; height: 200px; border: 2px dashed rgba(255,255,255,0.15);
    border-radius: 12px; margin: 0 auto; display: flex; align-items: center;
    justify-content: center; color: #555; font-size: 0.9rem;
  }
  .info { margin-top: 12px; font-size: 0.8rem; color: #666; }
  .error { color: #f5576c; margin-top: 12px; font-size: 0.9rem; }
  .loading { display: inline-block; width: 20px; height: 20px; border: 2px solid rgba(255,255,255,0.2);
    border-top-color: #f5576c; border-radius: 50%; animation: spin 0.8s linear infinite; margin-right: 8px; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="container">
  <h1>🎨 Flow Matching 手写数字生成</h1>
  <p class="subtitle">MNIST 条件生成 · V100 FP16 训练 · ODE 采样</p>

  <div class="digit-selector" id="digitSelector">
  </div>

  <div class="controls">
    <select id="methodSelect">
      <option value="rk4">RK4 (推荐 · ~50步)</option>
      <option value="euler">Euler (~100步)</option>
    </select>
    <button class="generate-btn" id="generateBtn" onclick="generate()">✨ 生成数字</button>
  </div>

  <div class="result-area" id="resultArea">
    <div class="placeholder" id="placeholder">👆 选择一个数字后点击生成</div>
    <img id="resultImage" style="display:none" alt="生成的手写数字">
  </div>
  <div class="info" id="info"></div>
  <div class="error" id="error"></div>
</div>

<script>
  // 初始化数字选择器
  const digitSelector = document.getElementById('digitSelector');
  let selectedDigit = 5;

  for (let i = 0; i < 10; i++) {
    const btn = document.createElement('button');
    btn.className = 'digit-btn' + (i === selectedDigit ? ' active' : '');
    btn.textContent = i;
    btn.onclick = () => selectDigit(i);
    digitSelector.appendChild(btn);
  }

  function selectDigit(d) {
    selectedDigit = d;
    document.querySelectorAll('.digit-btn').forEach((b, i) => {
      b.classList.toggle('active', i === d);
    });
  }

  async function generate() {
    const btn = document.getElementById('generateBtn');
    const placeholder = document.getElementById('placeholder');
    const img = document.getElementById('resultImage');
    const info = document.getElementById('info');
    const error = document.getElementById('error');
    const method = document.getElementById('methodSelect').value;
    const steps = method === 'rk4' ? 50 : 100;

    btn.disabled = true;
    btn.innerHTML = '<span class="loading"></span>生成中...';
    error.textContent = '';
    info.textContent = '';

    try {
      const resp = await fetch('/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ digit: selectedDigit, method, steps }),
      });
      const data = await resp.json();

      if (data.success) {
        placeholder.style.display = 'none';
        img.style.display = 'block';
        img.src = 'data:image/png;base64,' + data.image_base64;
        info.textContent = `数字 ${data.digit} · ${data.method.toUpperCase()} · ${data.steps} 步积分`;
      } else {
        error.textContent = '❌ ' + data.error;
      }
    } catch (e) {
      error.textContent = '❌ 网络错误: ' + e.message;
    } finally {
      btn.disabled = false;
      btn.innerHTML = '✨ 生成数字';
    }
  }

  // 页面加载后自动生成一次默认数字
  window.onload = () => generate();
</script>
</body>
</html>"""


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flow Matching MNIST Web 推理服务")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint 文件路径")
    parser.add_argument("--port", type=int, default=5000, help="服务端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="绑定地址")
    args = parser.parse_args()

    load_model(args.checkpoint)

    print(f"\n🌐 Web 服务启动: http://{args.host}:{args.port}")
    print(f"   按 Ctrl+C 停止服务\n")

    app.run(host=args.host, port=args.port, debug=False)
