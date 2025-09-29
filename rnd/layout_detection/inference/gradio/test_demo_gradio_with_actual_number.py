import gradio as gr
from PIL import Image
import numpy as np
import cv2
from inference import TiledInference
import os
import json

# ==== Model Configuration ====
MODEL_PATH = r"runs/train/experiment_20250714_173429/train/weights/best.pt"
REPORT_JSON_PATH = "report_output.json"

# ==== Load Ground Truth JSON ====
with open(REPORT_JSON_PATH, "r") as f:
    report_data = json.load(f)

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

    input_image = Image.open(input_image_path).convert("RGB")
    if model is None:
        return None, "❌ Model not initialized."

    try:
        img_array = np.array(input_image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        height, width = img_bgr.shape[:2]

        detections = model.predict_with_multiple_scales(img_bgr)
        result_img_bgr = model.draw_detections(img_bgr, detections)
        result_img_rgb = cv2.cvtColor(result_img_bgr, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(result_img_rgb)

        class_counts = {}
        for det in detections:
            cls = det['class_name']
            class_counts[cls] = class_counts.get(cls, 0) + 1

        total_detections = len(detections)
        count_text = f"🔍 Total Detections: {total_detections}\n"
        count_text += f"📏 Image Size: {width}x{height}\n"

        # Find ground truth from JSON
        matched_entry = None
        for entry in report_data:
            if entry['main name'] == filename:
                matched_entry = entry
                break

        class_mapping = {
            "B1": "B1",
            "B1/EM": "B1/EM",
            "D1": "D1",
            "EX": "Exit",
            "SM": "SM"
        }

        if matched_entry is None:
            # No match, print NA for all
            for det_class, _ in sorted(class_mapping.items()):
                pred_count = class_counts.get(det_class, 0)
                count_text += f"• {det_class}: {pred_count}/NA\n"
        else:
            for det_class, json_key in sorted(class_mapping.items()):
                print(det_class, json_key)
                pred_count = class_counts.get(det_class, 0)
                gt_count = matched_entry.get(json_key, "NA")
                if gt_count == 0:
                    gt_count = "NA"
                count_text += f"• {det_class}: {pred_count}/{gt_count}\n" ## B1 Right Predict / B1

        if detections:
            confs = [d['confidence'] for d in detections]
            count_text += f"\n📈 Confidence:\n• Avg: {np.mean(confs):.3f} | Max: {np.max(confs):.3f} | Min: {np.min(confs):.3f}\n"
        else:
            count_text += "\n❌ No objects detected"

        return result_pil, count_text

    except Exception as e:
        return None, f"❌ Error during prediction: {str(e)}"


# ==== Gradio Interface ====
info_markdown = """
### ⚡ Detected Electrical Components

The model is currently trained to detect the following symbols commonly found in electrical CAD layouts:

- **B1** – SURFACE MOUNTED LED BATTEN LUMINAIRE. 
- **B1/EM** – SURFACE MOUNTED LED BATTEN LUMINAIRE With Motion Sensor
- **D1** – SURFACE MOUNTED LED DOWNLIGHT LUMINAIRE.
- **EX** – Exit Sign or Emergency Exit Indicator  
- **SM** – 360° LIGHTING MOTION SENSOR. 
"""

gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="filepath", label="Upload Image"),
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
