import argparse
from prompt_cleaner import clean_prompt
from prompt_rewriter import create_dalle_prompt, generate_variations
from image_generator import generate_and_save_images, load_model
from originality_estimation import estimate_originality
from originality_estimation import load_clip_model
from final_output import find_and_copy_best_images

def main(api_key, input_prompt, keywords_csv_path, image_output_dir, final_output_dir, iterations, batches):
    # Cleaning the input prompt to remove any sensitive or copyrighted content
    clean_prompt_result = clean_prompt(api_key, input_prompt)
    print("Cleaned the input prompt.")

    # Generating multiple variations of the cleaned prompt for diverse image generation
    variations = generate_variations(api_key, clean_prompt_result, batches, iterations)
    print(f"Generated {len(variations)} variations of the clean prompt.")

    # Loading the image generation model
    pipe = load_model()
    print("Model loaded successfully.")

    # Generating images based on the rewritten prompts
    generate_and_save_images(variations, input_prompt, keywords_csv_path, image_output_dir, pipe)
    print("Sample images generated and saved.")

    # Estimating the originality of generated images to ensure uniqueness
    model, preprocess, device = load_clip_model()
    print("CLIP loaded successfully.")
    estimate_originality(image_output_dir, model, preprocess, device)
    print("Originality of images estimated.")

    # Identifying and copying the best image to the final output directory
    find_and_copy_best_images(image_output_dir, final_output_dir)
    print("Genericized image saved to the final output directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from prompts")
    parser.add_argument("--api_key", type=str, required=True, help="Your OpenAI API key")
    parser.add_argument("--input_prompt", type=str, required=True, help="The input character name for the prompt")
    parser.add_argument("--keywords_csv_path", type=str, required=True, help="Path to the negative keywords CSV file")
    parser.add_argument("--image_output_dir", type=str, required=True, help="Directory path for output images")
    parser.add_argument("--final_output_dir", type=str, required=True, help="Directory path for final output images")
    parser.add_argument("--iterations", type=int, required=True, help="Number of iterations per batch")
    parser.add_argument("--batches", type=int, required=True, help="Number of batches")

    args = parser.parse_args()

    main(args.api_key, args.input_prompt, args.keywords_csv_path, args.image_output_dir,
         args.final_output_dir, args.iterations, args.batches)
