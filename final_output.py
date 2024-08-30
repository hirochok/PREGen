import os
import pandas as pd
import shutil
from glob import glob

def find_and_copy_best_images(source_dir, final_output_dir):
    # Ensure the final output directory exists
    os.makedirs(final_output_dir, exist_ok=True)

    # Load the CSV file that contains the difference scores
    csv_path = os.path.join(source_dir, "average_clip_distance.csv")
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return  # If CSV doesn't exist, exit the function

    df = pd.read_csv(csv_path)

    # Find the image with the lowest "Difference Score"
    best_image_row = df.loc[df['Difference Score'].idxmin()]
    best_image_name = os.path.basename(best_image_row['Image Path'])

    # Copy the image with the lowest difference score to the final output folder
    src_image_path = os.path.join(source_dir, best_image_name)
    dst_image_name = "final_output.png"  # Fixed name for the best image, or use best_image_name for unique names
    dst_image_path = os.path.join(final_output_dir, dst_image_name)

    shutil.copy(src_image_path, dst_image_path)
    print(f"Copied {src_image_path} to {dst_image_path}")

#---Example usage---#
#source_dir = "path"
#final_output_dir = "path"
#find_and_copy_best_images(source_dir, final_output_dir)
