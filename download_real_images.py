import os
import requests
from tqdm import tqdm

OUTPUT_DIR = "dataset/real"
NUM_IMAGES = 200
UNSPLASH_ACCESS_KEY = "PUT_YOUR_ACCESS_KEY_HERE"

os.makedirs(OUTPUT_DIR, exist_ok=True)

headers = {
    "Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"
}

page = 1
saved = 0

with tqdm(total=NUM_IMAGES) as pbar:
    while saved < NUM_IMAGES:
        url = f"https://api.unsplash.com/photos?page={page}&per_page=30"
        response = requests.get(url, headers=headers).json()

        for photo in response:
            if saved >= NUM_IMAGES:
                break

            img_url = photo["urls"]["regular"]
            img_data = requests.get(img_url).content

            with open(os.path.join(OUTPUT_DIR, f"real_{saved:04d}.jpg"), "wb") as f:
                f.write(img_data)

            saved += 1
            pbar.update(1)

        page += 1

print(f"Downloaded {saved} real images.")
