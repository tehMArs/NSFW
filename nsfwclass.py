import asyncio
import aiohttp
import base64
from PIL import Image
import io
import json
import os
import glob
import shutil
import time

LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "olmocr-7b-0225-preview"
RESIZE_SIZE = (1024, 1024) # 2048, 2048
JPEG_QUALITY = 100
TEMPERATURE = 0.1
TOP_P = 0.95
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']
INPUT_DIRECTORY = "Y:\\Pictures"
OUTPUT_FILE = "image_classification_results.txt"
CUSTOM_PROMPT = "Classify this image as NSFW or SFW. Respond only 'NSFW' or 'SFW'."
NSFW_SUBFOLDER_NAME = "Y:\\NSFW_images"
SFW_SUBFOLDER_NAME = "Y:\\SFW_Images"
CONCURRENCY_LIMIT = 5

async def resize_and_encode_image(image_path, resize_dims, jpeg_quality, maintain_aspect_ratio=True):
    try:
        start_time = time.time()
        img = Image.open(image_path)
        img = img.convert("RGB")
        if maintain_aspect_ratio:
            original_width, original_height = img.size
            target_width, target_height = resize_dims
            aspect_ratio = original_width / original_height
            if original_width > original_height:
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
            else:
                new_height = target_height
                new_width = int(target_height * aspect_ratio)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            if (new_width, new_height) != resize_dims:
                left = (new_width - target_width) / 2
                top = (new_height - target_height) / 2
                right = (new_width + target_width) / 2
                bottom = (new_height + target_height) / 2
                img = img.crop((left, top, right, bottom))
        else:
            img = img.resize(resize_dims, Image.LANCZOS)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=jpeg_quality, optimize=True)
        img_byte_arr = img_byte_arr.getvalue()
        base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
        return base64_image
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

async def get_classification_from_api(base64_image_data, prompt, session):
    headers = {"Content-Type": "application/json"}
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_data}"}}
            ]}
        ],
        "max_tokens": 200,
        "stream": False,
        "temperature": TEMPERATURE,
        "top_p": TOP_P
    }

    try:
        async with session.post(LM_STUDIO_API_URL, headers=headers, json=data) as response:
            response.raise_for_status()
            json_response = await response.json()

        if 'choices' in json_response and json_response['choices']:
            full_response_text = json_response['choices'][0]['message']['content'].strip()
            description, extracted_classification = extract_classification_from_response(full_response_text)
            return extracted_classification, description, full_response_text
        else:
            print("  Debug (get_classification_from_api): No 'choices' in API response.") # Keep essential debug for API issues
            return None, None, None

    except aiohttp.ClientError as e:
        print(f"  Debug (get_classification_from_api): API request error: {e}") # Keep essential debug for API issues
        return None, None, None
    except json.JSONDecodeError:
        print(f"  Debug (get_classification_from_api): Error decoding API response.") # Keep essential debug for API issues
        return None, None, None
    except Exception as e:
        print(f"  Debug (get_classification_from_api): Unexpected error: {e}") # Keep essential debug for API issues
        return None, None, None

def extract_classification_from_response(full_response_text):
    description = ""
    classification = "Unclear"
    if "NSFW" in full_response_text.upper():
        classification = "NSFW"
        description_end_index = full_response_text.upper().rfind("NSFW")
        description = full_response_text[:description_end_index].strip()
    elif "SFW" in full_response_text.upper():
        classification = "SFW"
        description_end_index = full_response_text.upper().rfind("SFW")
        description = full_response_text[:description_end_index].strip()
    else:
        description = full_response_text
        print("  Debug: Classification keywords 'NSFW' or 'SFW' not found in response.") # Keep essential debug for keyword extraction issues
    return description, classification.strip().upper()

def combine_classifications(classification1, classification2):
    if classification1 == classification2:
        return classification1
    else:
        return "Unclear"

def choose_description(description1, description2):
    return description1

async def process_image_and_classify(image_path, prompt, session, semaphore):
    async with semaphore:
        base64_image_data = await resize_and_encode_image(image_path, RESIZE_SIZE, JPEG_QUALITY)
        if not base64_image_data:
            return None, "Error encoding image"

        # --- First API Call ---
        classification_result1, description1, full_response_text1 = await get_classification_from_api(base64_image_data, prompt, session)
        if classification_result1 is None:
            return None, "API call failed (first attempt)"

        # --- Second API Call (same image, same prompt) ---
        classification_result2, description2, full_response_text2 = await get_classification_from_api(base64_image_data, prompt, session)
        if classification_result2 is None:
            return None, "API call failed (second attempt)"

        # --- Combine Classifications ---
        final_classification = combine_classifications(classification_result1, classification_result2)
        final_description = choose_description(description1, description2)

        print(f"  Debug: Classification 1: '{classification_result1}', Classification 2: '{classification_result2}', Final Classification: '{final_classification}'") # Keep double prompt debug for now
        return final_description, final_classification


async def main():
    input_dir = INPUT_DIRECTORY
    output_filepath = OUTPUT_FILE
    prompt_to_use = CUSTOM_PROMPT
    nsfw_folder_name = NSFW_SUBFOLDER_NAME
    sfw_folder_name = SFW_SUBFOLDER_NAME
    model_name_to_use = MODEL_NAME
    concurrency_limit = CONCURRENCY_LIMIT

    if model_name_to_use == "your_vision_model_name":
        print("Error: Please set the MODEL_NAME variable in the script to your vision model's name in LM Studio.")
        exit(1)
    if input_dir == "path/to/your/image/directory":
        print("Error: Please set the INPUT_DIRECTORY variable in the script to the path of your image directory.")
        exit(1)

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' is not a valid directory.")
        exit(1)

    nsfw_output_dir = os.path.join(input_dir, nsfw_folder_name)
    os.makedirs(nsfw_output_dir, exist_ok=True)
    sfw_output_dir = os.path.join(input_dir, sfw_folder_name)
    os.makedirs(sfw_output_dir, exist_ok=True)

    image_files = []
    for ext_pattern in IMAGE_EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(input_dir, ext_pattern)))

    print(f"Debug: Found {len(image_files)} image files.")
    if not image_files:
        print(f"No image files found in directory '{input_dir}' with extensions: {', '.join(IMAGE_EXTENSIONS)}")
        exit(1)
    total_images = len(image_files)
    print(f"Processing {total_images} image files from '{input_dir}' using model '{model_name_to_use}' from '{LM_STUDIO_API_URL}'...")

    start_overall_time = time.time()
    semaphore = asyncio.Semaphore(concurrency_limit)

    async with aiohttp.ClientSession() as session:
        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            outfile.write("Filename\tDescription\tClassification\tMoved to NSFW Folder\tMoved to SFW Folder\n")
            for index, image_file in enumerate(image_files):
                print(f"Processing image {index + 1}/{total_images}: {image_file}")
                description, classification = await process_image_and_classify(image_file, prompt_to_use, session, semaphore)
                moved_to_nsfw = "No"
                moved_to_sfw = "No"
                print(f"  Classification: {classification}") # Keep essential classification output

                if description is not None:
                    print(f"  Description: {description[:80]}...") # Keep essential description output
                    if classification == "NSFW":
                        try:
                            destination_path = os.path.join(nsfw_output_dir, os.path.basename(image_file))
                            shutil.move(image_file, destination_path)
                            moved_to_nsfw = "Yes"
                            print(f"  Moved '{os.path.basename(image_file)}' to '{nsfw_folder_name}' folder.") # Keep essential move output
                        except Exception as move_err:
                            print(f"  Error moving file to NSFW folder: {move_err}") # Keep essential error output
                            moved_to_nsfw = f"Move Error: {move_err}"
                    elif classification == "SFW":
                        try:
                            destination_path = os.path.join(sfw_output_dir, os.path.basename(image_file))
                            shutil.move(image_file, destination_path)
                            moved_to_sfw = "Yes"
                            print(f"  Moved '{os.path.basename(image_file)}' to '{sfw_folder_name}' folder.") # Keep essential move output
                        except Exception as move_err:
                            print(f"  Error moving file to SFW folder: {move_err}") # Keep essential error output
                            moved_to_sfw = f"Move Error: {move_err}"
                    else:
                        print(f"  Debug (Main Loop - Unclear): Classification is not NSFW or SFW: '{classification}'") # Keep essential debug for unclear classification

                    outfile.write("{}\t{}\t{}\t{}\t{}\n".format(os.path.basename(image_file), description.replace('\t', ' ').replace('\n', ' '), classification, moved_to_nsfw, moved_to_sfw))
                else:
                    print(f"  Error: {classification}") # Keep essential error output
                    outfile.write("{}\tError: {}\tError\tError\tError\n".format(os.path.basename(image_file), classification))

    end_overall_time = time.time()
    overall_processing_time = end_overall_time - start_overall_time
    print(f"Processing complete. Results saved to '{output_filepath}'. Total processing time: {overall_processing_time:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
