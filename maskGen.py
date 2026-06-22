import numpy as np
from PIL import Image
import os
from tqdm import tqdm

def generate_mask_random_loc(patch_size, num_patches):
    
    mask = np.zeros((28,28))

    for _ in range(num_patches):
        x = np.random.randint(0, 28 - patch_size)
        y = np.random.randint(0, 28 - patch_size)

        noise_patch = np.ones((patch_size, patch_size))

        mask[x:x+patch_size, y:y+patch_size] = noise_patch

    mask = (mask*255).astype(np.uint8)

    return mask

def save_dataset(folder, num_samples):
    os.makedirs(folder, exist_ok=True)

    for i in tqdm(range(num_samples), desc=f"Generating {folder}"):
        mask = generate_mask_random_loc(np.random.randint(1,14), np.random.randint(1,5))
        mask_pil = Image.fromarray(mask)
        mask_pil.save(os.path.join(folder, f"{i:06d}_mask.png"))

save_dataset("data/mask/train", 60000)
save_dataset("data/mask/test", 10000)