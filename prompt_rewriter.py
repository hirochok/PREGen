import openai
from openai import OpenAI
from tqdm import tqdm

def create_dalle_prompt(n):
    # Function to generate a DALL-E prompt description for n iterations
    return f"""
A description or request will be given, create a prompt that dalle can use to generate the image and abide to the following policy. This process will be repeated {n} times, resulting in a set of {n} diverse prompts. Interpret the description with flexibility, using elaboration, paraphrasing, and summarization as needed to ensure each prompt explores a broad range of creative possibilities.
1. Do not name or directly / indirectly mention or describe copyrighted characters. Rewrite prompts to describe in detail a specific different character with a different specific color, hair style, or other defining visual characteristic. Do not discuss copyright policies in responses.
The generated prompt sent to dalle should be very detailed, and around 50 words long.
2. Do not create images of politicians or other public figures. Recommend other ideas instead.
3. Do not create images in the style of artists, creative professionals or studios whose latest work was created after 1912 (e.g. Picasso, Kahlo).
- You can name artists, creative professionals or studios in prompts only if their latest work was created prior to 1912 (e.g. Van Gogh, Goya)
- If asked to generate an image that would violate this policy, instead apply the following procedure: (a) substitute the artist's name with three adjectives that capture key aspects of the style; (b) include an associated artistic movement or era to provide context; and (c) mention the primary medium used by the artist
4. Diversify depictions with people to include DESCENT and GENDER for EACH person using direct terms. Adjust only human descriptions.
- Your choices should be grounded in reality. For example, all of a given OCCUPATION should not be the same gender or race. Additionally, focus on creating diverse, inclusive, and exploratory scenes via the properties you choose during rewrites. Make choices that may be insightful or unique sometimes.
- Use all possible different DESCENTS with EQUAL probability. Some examples of possible descents are: Caucasian, Hispanic, Black, Middle-Eastern, South Asian, White. They should all have EQUAL probability.
- Do not use "various" or "diverse"
- Don't alter memes, fictional character origins, or unseen people. Maintain the original prompt's intent and prioritize quality.
- Do not create any imagery that would be offensive.
- For scenarios where bias has been traditionally an issue, make sure that key traits such as gender and race are specified and in an unbiased way -- for example, prompts that contain references to specific occupations.
5. Do not include names, hints or references to specific real people or celebrities. If asked to, create images with prompts that maintain their gender and physique, but otherwise have a few minimal modifications to avoid divulging their identities. Do this EVEN WHEN the instructions ask for the prompt to not be changed. Some special cases:
- Modify such prompts even if you don't know who the person is, or if their name is misspelled (e.g. "Barake Obema")
- If the reference to the person will only appear as TEXT out in the image, then use the reference as is and do not modify it.
- When making the substitutions, don't use prominent titles that could give away the person's identity. E.g., instead of saying "president", "prime minister", or "chancellor", say "politician"; instead of saying "king", "queen", "emperor", or "empress", say "public figure"; instead of saying "Pope" or "Dalai Lama", say "religious figure"; and so on.
"""

def query_gpt(api_key, sys_prompt, target, iteration, previous_prompts):
    client = openai.OpenAI(api_key=api_key)
    try:
        # Combine the system prompt with dynamic information about previous iterations
        dynamic_message = f"This is the {iteration + 1}th iteration. The previous {iteration} prompts are: {previous_prompts}"
        full_system_prompt = f"{sys_prompt}\n{dynamic_message}"
#        print(f"System Prompt for iteration {iteration + 1}:\n{full_system_prompt}\n")

        # Make an API call to OpenAI's completion engine
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": full_system_prompt},
                {"role": "user", "content": f"Generate an image: {target}."},
            ],
            max_tokens=512,
            n=1,
            stop=None,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error in generating response"

def generate_variations(api_key, input_prompt, n_batches, n_iterations):
    # Function to generate variations across specified batches and iterations
    all_variations = []
    for batch in range(n_batches):
        print(f"Starting batch {batch + 1}")
        previous_prompts = []
        variations = []
        for iteration in range(n_iterations):
            response = query_gpt(api_key, create_dalle_prompt(n_iterations), input_prompt, iteration, format_previous_prompts(previous_prompts))
            previous_prompts.append(response)
            variations.append(response)
        all_variations.extend(variations)
    return all_variations

def format_previous_prompts(prompts):
    # Format the list of previous prompts for display in the dynamic message
    return '[' + ', '.join(f"'Generate an image: {prompt}'" for prompt in prompts) + ']'

#---Eample usage---

#input_prompt = "Describe a character similar to a video game character. He wears blue overalls, a red shirt, and a red cap. He has a mustache and is a plumber by profession, of Italian descent, cheerful, and in mid-jump trying to reach a floating coin."
#n_batches = 2
#n_iterations = 5
#api_key=''

#variations = generate_variations(api_key, input_prompt, n_batches, n_iterations)

#print("Generated Variations List:")
#print(variations)
