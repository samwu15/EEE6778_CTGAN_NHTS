# ui/app.py — robust local desktop version
import sys
from pathlib import Path

import torch
import gradio as gr
from PIL import Image
from torchvision import transforms

# --- 專案根目錄 & 路徑 ---
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cyclegan_min import GeneratorResnet  # ✅ 你原本的模型

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

DEFAULT_CKPT = ROOT / "result1" / "checkpoints" / "cyclegan_ultra_epoch_01.pt"

# ---------------------------
# 1. 載入模型（這段請你依照你原本的版本微調）
# ---------------------------

_G_AB = None          # global generator
_TRANSFORM = None     # global preprocess
_DENORM_MEAN = [0.5, 0.5, 0.5]
_DENORM_STD = [0.5, 0.5, 0.5]


def _build_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def _load_model():
    global _G_AB, _TRANSFORM

    state = torch.load(DEFAULT_CKPT, map_location=DEVICE)

    # 從 checkpoint 拿 meta
    img_size = int(state.get("img_size", 64))
    n_res = int(state.get("n_res", 2))

    # 建立模型（這裡依你 cyclegan_min.GeneratorResnet 的參數調）
    # 如果你的 __init__ 長得不一樣，請只改這一行
    _G_AB = GeneratorResnet(n_res=n_res).to(DEVICE)

    # checkpoint 內容可能是：
    # 1) {"G_AB": state_dict, "G_BA": ... , "img_size": ..., "n_res": ...}
    # 2) 或直接是 state_dict
    if "G_AB" in state:
        G_AB_state = state["G_AB"]
    else:
        G_AB_state = state

    _G_AB.load_state_dict(G_AB_state)
    _G_AB.eval()

    _TRANSFORM = _build_transform(img_size)

    print(f"[INFO] Model loaded. img_size={img_size}, n_res={n_res}, device={DEVICE}")


# 啟動時就先載入一次
_load_model()


# ---------------------------
# 2. 推論函數（CPU 版優化）
# ---------------------------

def translate_image(pil_img: Image.Image) -> Image.Image:
    """把上傳的圖片 → 轉成電影風格圖片"""
    if pil_img is None:
        return None

    # 前處理
    x = _TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():  # ✅ 推論不需要梯度，省很多時間
        if USE_AMP:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                y = _G_AB(x)
        else:
            y = _G_AB(x)

    # 後處理：從 [-1, 1] 還原回 [0, 255]
    y = y[0].detach().cpu()
    for c in range(3):
        y[c] = y[c] * _DENORM_STD[c] + _DENORM_MEAN[c]
    y = torch.clamp(y, 0.0, 1.0)

    y_np = (y.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(y_np)


# ---------------------------
# 3. Gradio UI（Deliverable 3 完整版）
# ---------------------------

def build_demo():
    with gr.Blocks(title="Ancient Painting → Film Style (CycleGAN)") as demo:
        gr.Markdown("# Ancient Painting → Film Style Translation")
        gr.Markdown(
            "上傳一張古畫風格圖片，模型會把它轉換成電影風格影像。\n"
            "這個介面是 Deliverable 3 的最終雛型。"
        )

        gr.Markdown(f"**Device:** `{DEVICE}` &nbsp;&nbsp; **Using AMP:** `{USE_AMP}`")
        gr.Markdown(f"**Checkpoint:** `{DEFAULT_CKPT}`")

        with gr.Row():
            with gr.Column():
                input_img = gr.Image(
                    label="輸入圖片（古畫）",
                    type="pil"
                )

                gr.Markdown("如果沒有圖片，可以先用一張範例圖：")

                # ⚠️ 這邊請你把路徑改成你實際有的一張圖片，例如 A 裡面的一張
                SAMPLE_PATH = ROOT / "A" / "sample_01.jpg"

                def load_sample():
                    if SAMPLE_PATH.exists():
                        return Image.open(SAMPLE_PATH)
                    else:
                        # 如果找不到，維持空白
                        return None

                sample_btn = gr.Button("載入範例圖片")
                sample_btn.click(fn=load_sample, outputs=input_img)

                clear_btn = gr.Button("清空輸入")
                clear_btn.click(fn=lambda: None, outputs=input_img)

            with gr.Column():
                output_img = gr.Image(
                    label="輸出圖片（電影風格）",
                    type="pil"
                )
                status = gr.Markdown("狀態：🟢 就緒")

                def wrapped_translate(img):
                    if img is None:
                        return None, "狀態：⚠️ 請先上傳或載入一張圖片"
                    out = translate_image(img)
                    return out, "狀態：✅ 完成推論"

                run_btn = gr.Button("開始轉換", variant="primary")
                run_btn.click(
                    fn=wrapped_translate,
                    inputs=input_img,
                    outputs=[output_img, status]
                )

        return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.queue()   # 保留 queue，之後你可以在報告說有排隊機制
    demo.launch()


