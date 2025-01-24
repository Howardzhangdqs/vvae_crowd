import os
import cv2
import json
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from multiprocessing import Pool, cpu_count

# Regular expression to match JSON files with numeric names
json_file_reg = re.compile(r'^\d+.json$')


# Function to generate a density map from given points
def generate_density_map(image_size, points, sigma=15):
    density_map = np.zeros(image_size, dtype=np.float32)
    h, w = image_size

    # Place a 1 at each point location
    for point in points:
        x, y = int(point[0]), int(point[1])
        if x < w and y < h:
            density_map[y, x] = 1

    # Apply Gaussian filter to create the density map
    density_map = gaussian_filter(density_map, sigma=sigma)
    density_map = density_map / density_map.sum() * len(points)
    return density_map


# Save the density map as an image
def save_density_map(density_map, output_path):
    plt.imsave(output_path, density_map, cmap='hot')


# Process a single JSON file and generate the corresponding density map
def process_json(json_path, image_size=(1080, 1920)):
    with open(json_path, 'r') as f:
        data = json.load(f)

    points = []
    # Extract points from the JSON data
    for key, value in data.items():
        regions = value['regions']
        for region in regions:
            shape = region['shape_attributes']
            x = shape['x'] + shape['width'] // 2
            y = shape['y'] + shape['height'] // 2
            points.append((x, y))

    # Generate and save the density map
    density_map = generate_density_map(image_size, points)
    output_path = json_path.replace('.json', '.png')
    save_density_map(density_map, output_path)


# Function to process all JSON files in a dataset directory
def process_dataset(dataset_dir):
    dataset_name = os.path.basename(dataset_dir)
    json_files = []
    # Collect all JSON files in the directory
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if json_file_reg.match(file):
                json_files.append(os.path.join(root, file))

    json_files.sort()

    # Use multiprocessing to process JSON files in parallel
    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap(process_json, json_files), total=len(json_files), desc=f"Processing {dataset_name}"))


# Process all datasets in the given path
if __name__ == "__main__":
    dataset_path = "/root/autodl-fs/FDST"
    datasets = os.listdir(dataset_path)
    for dataset in datasets:
        dataset_dir = os.path.join(dataset_path, dataset)
        process_dataset(dataset_dir)
