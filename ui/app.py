import gradio as gr
from PIL import Image
import torch

# 載入你訓練好的模型
model = torch.load("results/cyclegan_model.pth", map_location="cpu")

def translate(image):
    # image -> tensor -> model -> output -> image
    with torch.no_grad():
        output = model(image)
    return output

demo = gr.Interface(
    fn=translate,
    inputs=gr.Image(type="pil"),
    outputs="image",
    title="Ancient-to-Film GAN",
    description="Upload an ancient painting to see its film-style transformation."
)

demo.launch()
