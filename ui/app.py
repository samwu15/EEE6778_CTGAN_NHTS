# ui/app.py — robust local desktop version
import sys
from pathlib import Path
from typing import Tuple
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# --- 將專案 src/ 加入 Python 路徑 ---
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import gradio as gr
from PIL import Image
from torchvision import transforms

from cyclegan_min import GeneratorResnet

# ---- 路徑與裝置設定（你目前放在 result1/）----
DEFAULT_CKPT = ROOT / "result1" / "checkpoints" / "cyclegan_ultra_epoch_01.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = (DEVICE.type == "cuda")  # 只有 CUDA 才啟用 AMP

def _build_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

def _denorm(x: torch.Tensor) -> torch.Tensor:
    return (x * 0.5 + 0.5).clamp(0, 1)

@torch.inference_mode()
def load_model(ckpt_path: Path = DEFAULT_CKPT) -> Tuple[torch.nn.Module, int]:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到模型檔：{ckpt_path}")
    state = torch.load(ckpt_path, map_location=DEVICE)

    img_size = int(state.get("img_size", 64))
    n_res = int(state.get("n_res", 2))

    G_AB = GeneratorResnet(n_res=n_res).to(DEVICE)

    # ===== 取出並修正權重鍵，寬鬆載入（解決 Unexpected keys / size mismatch）=====
    cand = None
    for key in ("G_AB", "generator_ab", "model", "state_dict"):
        if key in state and isinstance(state[key], dict):
            cand = state[key]
            break
    if cand is None:
        cand = state if isinstance(state, dict) else {}

    # 有些 checkpoint 權重帶 "model." 前綴 → 去掉
    if any(isinstance(k, str) and k.startswith("model.") for k in cand.keys()):
        cand = {k.replace("model.", "", 1): v for k, v in cand.items()}

    missing, unexpected = G_AB.load_state_dict(cand, strict=False)
    if unexpected:
        print("⚠️  Unexpected keys ignored:", list(unexpected)[:6], "...")
    if missing:
        print("⚠️  Missing keys (kept random init):", list(missing)[:6], "...")

    G_AB.eval()
    print(f"✅ G_AB loaded on {DEVICE} | img_size={img_size}")
    return G_AB, img_size

# --- 全域快取 ---
_G_AB = None
_IMG_SIZE = None
_TFM = None

def _ensure_model():
    global _G_AB, _IMG_SIZE, _TFM
    if _G_AB is None:
        print("📂 Loading checkpoint from:", DEFAULT_CKPT)
        _G_AB, _IMG_SIZE = load_model(DEFAULT_CKPT)
        _TFM = _build_transform(_IMG_SIZE)

def translate_image(pil_img: Image.Image) -> Image.Image:
    _ensure_model()
    x = _TFM(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    # 非 CUDA 時關掉 AMP，避免 dtype 問題
    if USE_AMP:
        with torch.amp.autocast("cuda", enabled=True):
            y = _G_AB(x)
    else:
        y = _G_AB(x)
    y = _denorm(y[0]).permute(1, 2, 0).detach().cpu().numpy()
    return Image.fromarray((y * 255).astype("uint8"))

def app():
    title = "Ancient → Film Style (CycleGAN Demo)"
    description = (
        "上傳一張古畫風格圖像（domain A），模型將輸出電影風格圖像（domain B）。"
        f"<br>Device: <b>{DEVICE.type.upper()}</b> | Using AMP: <b>{USE_AMP}</b>"
        f"<br>Checkpoint: <code>{DEFAULT_CKPT.relative_to(ROOT)}</code>"
    )
    return gr.Interface(
        fn=translate_image,
        inputs=gr.Image(type="pil", label="Upload ancient-style image (A)"),
        outputs=gr.Image(type="pil", label="Generated film-style image (B)"),
        title=title,
        description=description,
        allow_flagging="never",
    )

if __name__ == "__main__":
    app().queue().launch(server_name="127.0.0.1", server_port=7860)

