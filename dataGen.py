import numpy as np
from scipy.stats import multivariate_normal
from PIL import Image
import os
from tqdm import tqdm

def generate_random_loc_gaussian(height, width, num_blobs):
    
    x = np.linspace(0, width-1, width)
    y = np.linspace(0, height-1, height)

    X,Y = np.meshgrid(x,y)

    pos = np.dstack((X,Y))
    image = np.zeros((height, width))

    for _ in range(num_blobs):
        mean = np.random.uniform(low=0, high=height-1, size=2)
        A = np.random.rand(2,2)
        cov = np.dot(A,A.T) + 15*np.eye(2)
        rv = multivariate_normal(mean=mean, cov=cov)
        image += rv.pdf(pos)

    image /= np.max(image)
    image = (image*255).astype(np.uint8)

    # import matplotlib.pyplot as plt
    # plt.imshow(image)
    # plt.show()
    return image
    
def save_dataset(folder, num_samples):
    os.makedirs(folder, exist_ok=True)

    for i in tqdm(range(num_samples), desc=f"Generating {folder}"):
        img = generate_random_loc_gaussian(28, 28, np.random.randint(20))
        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(folder, f"{i:06d}.png"))

save_dataset("data/images/train", 60000)
save_dataset("data/images/test", 10000)