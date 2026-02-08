import json
import torch
import numpy as np
import random
from PIL import Image
from torchvision import models, transforms

# ---------- config ----------
with open("models/layer_b_config.json") as f:
    CONFIG = json.load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- load model ----------
def load_model():
    # EXACT model used in training
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)

    state_dict = torch.load(
        "models/layer_b_model.pt",
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)  # strict=True by default
    model.to(DEVICE)
    model.eval()
    return model


MODEL = load_model()

# ---------- transforms ----------
TRANSFORM = transforms.Compose([
    transforms.Resize((CONFIG["input_size"], CONFIG["input_size"])),
    transforms.ToTensor(),
    # transforms.Normalize(CONFIG["mean"], CONFIG["std"])
])

# ---------- patch sampling ----------
def sample_random_patch(image, patch_size):
    w, h = image.size

    if w < patch_size or h < patch_size:
        return image.resize((patch_size, patch_size))

    x = random.randint(0, w - patch_size)
    y = random.randint(0, h - patch_size)
    return image.crop((x, y, x + patch_size, y + patch_size))


# ---------- inference ----------
def infer_image(image_path):
    image = Image.open(image_path).convert("RGB")
    probs = []

    with torch.no_grad():
        for _ in range(CONFIG["num_patches"]):
            patch = sample_random_patch(image, CONFIG["input_size"])
            patch = TRANSFORM(patch).unsqueeze(0).to(DEVICE)

            logit = MODEL(patch)
            prob = torch.sigmoid(logit).item()
            probs.append(prob)

    probs = np.array(probs)

    if CONFIG["aggregation"] == "percentile":
        score = np.percentile(probs, CONFIG["percentile"])
    else:
        score = probs.mean()

    return {
        "confidence": float(score),
        "is_ai_generated": bool(score >= 0.5)
    }
