
import gradio as gr
from PIL import Image
import numpy as np

def infer_fn(img):
    # Demo 階段先直接回傳原圖；之後換成 load_and_infer()
    return Image.fromarray(np.uint8(img))

demo = gr.Interface(
    fn=infer_fn,
    inputs=gr.Image(type="numpy", label="Upload Ancient Drawing"),
    outputs=gr.Image(label="Film-Style Output"),
    title="Ancient → Film GAN Demo",
    description="Upload an ancient drawing to see film-style conversion (demo mode)."
)

if __name__ == "__main__":
    demo.launch()

