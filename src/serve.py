import io
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from PIL import Image


from model import get_model

app = Flask(__name__)

CHECKPOINT_PATH = Path("/app/checkpoints/classifier_v1.pt")
if not CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH = Path("checkpoints/classifier_v1.pt")

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

device = torch.device("cpu")   # serving runs on CPU; no training happening here
model = get_model(architecture="simple_cnn", num_classes=10).to(device)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()   # inference mode — disables Dropout, freezes BatchNorm stats

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((32, 32))
    array = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    tensor = array.reshape(32, 32, 3).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - MEAN) / STD
    return tensor.unsqueeze(0)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None}), 200


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No 'image' file provided"}), 400

    file = request.files["image"]
    image_bytes = file.read()

    try:
        input_tensor = preprocess_image(image_bytes).to(device)
    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    predicted_idx = int(torch.argmax(probabilities))
    result = {
        "predicted_class": CLASS_NAMES[predicted_idx],
        "confidence": round(float(probabilities[predicted_idx]), 4),
        "all_probabilities": {
            CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(probabilities)
        },
    }
    return jsonify(result), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)