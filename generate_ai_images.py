import os
import random
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# ---------------- CONFIG ----------------
OUTPUT_DIR = "dataset/ai"
NUM_IMAGES = 200

# Choose: "last" or "random"
SAMPLING_MODE = "last"   # change to "random" if needed
RANDOM_SEED = 42
# ----------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading Nano-Banana dataset (from cache if available)...")

# This WILL NOT re-download if already cached
dataset = load_dataset(
    "bitmind/nano-banana",
    split="train",
)

total = len(dataset)
print(f"Dataset size: {total}")

if total < NUM_IMAGES:
    raise ValueError("Dataset smaller than requested sample size")

# -------- Sampling logic --------
if SAMPLING_MODE == "last":
    indices = list(range(total - NUM_IMAGES, total))

elif SAMPLING_MODE == "random":
    random.seed(RANDOM_SEED)
    indices = random.sample(range(total), NUM_IMAGES)

else:
    raise ValueError("SAMPLING_MODE must be 'last' or 'random'")

# -------- Save images only --------
saved = 0

for idx in tqdm(indices, desc="Saving images"):
    item = dataset[idx]

    # Dataset stores PIL images already
    image = item["image"]

    if not isinstance(image, Image.Image):
        continue

    filename = f"ai_{saved:04d}.png"
    image.save(os.path.join(OUTPUT_DIR, filename))
    saved += 1

print(f"✅ Done. Saved {saved} AI-generated images to '{OUTPUT_DIR}'")
