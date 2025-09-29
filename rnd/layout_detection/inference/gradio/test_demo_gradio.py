import gradio as gr
from PIL import Image
import numpy as np
import cv2
from inference import TiledInference
import os

# ==== Model Configuration ====
MODEL_PATH = r"runs/train/experiment_20250714_173429/train/weights/best.pt"



# ==== Initialize model ====
def initialize_model():
    try:
        model = TiledInference(
            model_path=MODEL_PATH,
            tile_size=1280,
            overlap=0.2,
            conf_threshold=0.8,
            nms_threshold=0.25
        )
        return model
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return None

model = initialize_model()

# ==== Prediction Function ====
def predict_image(input_image_path):

    filename = os.path.basename(input_image_path)
    print("filename:", filename)

    input_image = Image.open(input_image_path).convert("RGB")
    if model is None:
        return None, "❌ Model not initialized."

    if input_image is None:
        return None, "❌ Please upload an image."

    try:
        img_array = np.array(input_image)
        if len(img_array.shape) == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array

        height, width = img_bgr.shape[:2]

        # ✅ Use multi-scale prediction for better results
        detections = model.predict_with_multiple_scales(img_bgr)

        result_img_bgr = model.draw_detections(img_bgr, detections)
        result_img_rgb = cv2.cvtColor(result_img_bgr, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(result_img_rgb)

        total_detections = len(detections)
        class_counts = {}
        for det in detections:
            cls = det['class_name']
            class_counts[cls] = class_counts.get(cls, 0) + 1

        count_text = f"🔍 Total Detections: {total_detections}\n📏 Image Size: {width}x{height}\n"
        for k, v in sorted(class_counts.items()):
            count_text += f"• {k}: {v}\n"

        if detections:
            confs = [d['confidence'] for d in detections]
            count_text += f"\n📈 Confidence:\n• Avg: {np.mean(confs):.3f} | Max: {np.max(confs):.3f} | Min: {np.min(confs):.3f}\n"
        else:
            count_text += "❌ No objects detected"

        return result_pil, count_text

    except Exception as e:
        return None, f"❌ Error during prediction: {str(e)}"


# Markdown info section about detectable symbols
info_markdown = """
### ⚡ Detected Electrical Components

The model is currently trained to detect the following symbols commonly found in electrical CAD layouts:

- **B1** – SURFACE MOUNTED LED BATTEN LUMINAIRE. 
- **B1/EM** – SURFACE MOUNTED LED BATTEN LUMINAIRE With Motion Sensor
- **D1** – SURFACE MOUNTED LED DOWNLIGHT LUMINAIRE.
- **EX** – Exit Sign or Emergency Exit Indicator  
- **SM** – 360° LIGHTING MOTION SENSOR. 

These components are typically used in electrical layout plans for buildings.
"""

# ==== Gradio Interface ====
gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="filepath", label="Upload Image"),
    #inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[
        gr.Image(type="pil", label="Predicted Image"),
        gr.Textbox(label="Detection Summary")
    ],
    title="Electrical Component Counter in CAD Design",
    description="Upload an image to detect and count electrical components in CAD drawings.",
    article=info_markdown
).launch(
    server_name="0.0.0.0",
    server_port=7600,
    share=True
)



# 🔍 Total Detections: 25
# 📏 Image Size: 656x1016
# • B1: 5/(N/A)
# • B1/EM: 5/24
# • D1: 5/4
# • EX: 5/5
# • SM: 5/5

# 📈 Confidence:
# • Avg: 0.938 | Max: 0.955 | Min: 0.885