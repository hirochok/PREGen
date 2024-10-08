import os
import pandas as pd
from tqdm import tqdm
from diffusers import DiffusionPipeline
import torch
from PIL import Image

def load_model():
# Initialize and load the model
    pipe = DiffusionPipeline.from_pretrained(
        "playgroundai/playground-v2.5-1024px-aesthetic",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")
    return pipe

def generate_and_save_images(variations, original_prompt, keywords_csv_path, output_dir, pipe):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load keywords from CSV
    df_keywords = pd.read_csv(keywords_csv_path)

    target = original_prompt  # Assuming original_prompt is a string like "Mario"

    for idx, prompt in enumerate(tqdm(variations)):
        try:
            # Fetch negative keywords for the target, with error handling if not found
            if target in df_keywords['target'].values:
                keywords_row = df_keywords[df_keywords['target'] == target]
                negative_prompt = f"{target}, {keywords_row['prompt'].values[0]}"
            else:
                negative_prompt = "Copyrighted character"  # Default if no keywords are available
                print(f"No negative prompts found for {target}, using default.")

            # Generate image
            result = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=50, guidance_scale=3)
            image = result.images[0]

            # Ensure image is in PIL format for saving
            if not isinstance(image, Image.Image):
                image = Image.fromarray((image * 255).astype(np.uint8))

            # Save image
            image_path = os.path.join(output_dir, f"image_{idx+1}.png")
            image.save(image_path)
            print(f"Saved image: {image_path}")

        except Exception as e:
            print(f"Error occurred while processing prompt for {target}: {e}")

#---Example usage---
#variations = ['"Create an image of a cheerful Italian plumber character in mid-jump, striving to reach a floating coin. He sports blue overalls, a vibrant red shirt, a matching red cap, and a well-groomed mustache. He embodies the spirit of adventure and determination."', '"Create an image of a jovial character of Italian descent, dressed in blue overalls paired with a bright red shirt and cap, sporting a neat mustache. He is a plumber by profession and is portrayed in the middle of a high jump, extending his hand to touch a floating coin."', 'Generate an image: "Visualize an Italian plumber who exudes a joyful persona. He\'s captured in the midst of an energetic leap, reaching out for a hovering coin. His attire consists of blue work overalls, a bright red shirt, and a red cap. His face is distinguished by a neat, well-groomed mustache."', 'Generate an image: "Depict a lively plumber character of Italian descent in action. Dressed in blue overalls, a fiery red shirt and a matching cap, his face features a well-kept mustache. This jubilant character is caught mid-jump, his hand outstretched towards a levitating coin."', 'Generate an image: "Imagine a character, a plumber of Italian heritage, known for his vibrant personality. He\'s attired in blue overalls, a red shirt, and a matching cap, and is distinguished by a well-maintained mustache. The scene captures him in a dynamic leap, his hand extended towards a floating coin."']

#original_prompt = "Mario"  # This could be the character name or a descriptor

#keywords_csv_path = "path/50keywords_laion2b_top5.csv"
#output_dir = "path"

#pipe = load_model()
#generate_and_save_images(variations, original_prompt, keywords_csv_path, output_dir, pipe)
