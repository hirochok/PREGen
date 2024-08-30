import clip
import torch
from PIL import Image
import glob
import pandas as pd
from multiprocessing.pool import ThreadPool
import os
from tqdm import tqdm

def load_clip_model():
    # Check if GPU is available and set the device accordingly
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load the CLIP model
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def preprocess_image(image_path, preprocess, device):
    img = Image.open(image_path).convert('RGB')
    return preprocess(img).unsqueeze(0).to(device)

def compute_similarity(model, image_tensor1, image_tensor2):
    with torch.no_grad():
        image_features1 = model.encode_image(image_tensor1)
        image_features2 = model.encode_image(image_tensor2)
        similarity = torch.nn.functional.cosine_similarity(image_features1, image_features2, dim=1)
        return similarity

def estimate_originality(folder_path, model, preprocess, device):
    image_paths = sorted(glob.glob(f'{folder_path}/*.png'))
    num_images = len(image_paths)

    if num_images == 0:
        print(f"No images found in {folder_path}")
        return

    images = [preprocess_image(path, preprocess, device) for path in image_paths]
    images_tensor = torch.cat(images, dim=0)

    difference_scores = []

    for i in tqdm(range(num_images), desc=f"Processing {folder_path}"):
        target_image = images_tensor[i:i+1]
        other_images = torch.cat([images_tensor[:i], images_tensor[i+1:]], dim=0)
        similarities = compute_similarity(model, target_image, other_images)
        avg_similarity = torch.mean(similarities)
        difference_score = 1 - avg_similarity.item()
        difference_scores.append(difference_score)

    # Save difference scores to CSV
    results_df = pd.DataFrame({
        'Image Path': [os.path.basename(path) for path in image_paths],
        'Difference Score': difference_scores
    })

    output_csv_path = os.path.join(folder_path, "average_clip_distance.csv")
    results_df.to_csv(output_csv_path, index=False)
    print(f"Saved results to {output_csv_path}")

#---Example usage---
#base_folder_path = "path"
#model, preprocess, device = load_clip_model()
#estimate_originality(base_folder_path, model, preprocess, device)
