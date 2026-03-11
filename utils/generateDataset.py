'''
Utility script to generate compressed images
For each image in the train, validate and test set, randomly pick
a value between 0.1 and 0.2 as the target bpp, and compress to 
that value
Store compression and bpp information in csv file
'''

import os
import csv
import random
import numpy as np
from PIL import Image
import glymur

splits = ["train", "validate", "test"]

for split in splits:
    gt_dir = os.path.join(split, "ground_truth")
    comp_dir = os.path.join(split, "compressed")
    
    csv_path = os.path.join(split, "compression_stats.csv")
    
    
    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "target_bpp", "compression_ratio"])
        
        if not os.path.exists(gt_dir):
            print(f"  -> Directory {gt_dir} not found. Skipping...")
            continue
            
        for filename in os.listdir(gt_dir):
            if not filename.endswith(".png"):
                continue
                
            img_path = os.path.join(gt_dir, filename)
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img)
            
            target_bpp = random.uniform(0.1, 0.2)
            compression_ratio = 24 / target_bpp
            
            out_filename = filename.replace(".png", ".jp2")
            out_path = os.path.join(comp_dir, out_filename)
            
            jp2 = glymur.Jp2k(
                out_path, 
                data=img_array, 
                cratios=[compression_ratio]
            )
            
            writer.writerow([out_filename, round(target_bpp, 4), round(compression_ratio, 2)])