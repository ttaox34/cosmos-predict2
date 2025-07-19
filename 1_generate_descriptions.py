# Save this as 1_generate_descriptions.py
import argparse
import json
import sys
from pathlib import Path

try:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    import cv2
    from tqdm import tqdm
except ImportError as e:
    sys.exit(f"ImportError: {e}. Please run pip install torch transformers==4.51.3 accelerate 'qwen-vl-utils[decord]' opencv-python tqdm")

VIDEO_DESCRIPTION_PROMPT = """
You are a professional video scene analyst and creative prompt writer. Your goal is to watch the user's input video and generate a high-quality, detailed English prompt that accurately and vividly describes the scene. This generated prompt will be used to guide a video generation model.

**Task Requirements:**

1.  Your primary task is to meticulously analyze the provided video and generate a comprehensive and visually rich description based solely on the visual information.
2.  Describe in detail the characteristics of the main subject(s), including their appearance, actions, expression, quantity, ethnicity, and posture. Also, describe the surrounding environment, overall style, spatial relationships, and camera angles.
3.  The entire output must be in English. If there is any text visible in the video (e.g., on signs, clothing, or book titles), retain it in its original form within quotes.
4.  You must carefully analyze the visual style of the video frames (e.g., cinematic, anime, photorealistic, documentary, vintage film) and incorporate a clear style description into the prompt.
5.  Describe any visible or implied movement. Even if the subject is static, describe their posture in a way that suggests potential action or the state they are in.
6.  Your output should convey natural movement attributes. Use simple and direct verbs to describe actions or potential actions related to the subject.
7.  Your description must be grounded in the visual evidence from the video frames. Pay close attention to details such as character clothing, accessories, background elements, lighting, and textures.
8.  Control the generated prompt to be around 80-100 words to ensure it is concise yet descriptive.

**Example of the generated English prompt:**

1.  A Japanese fresh film-style video of a young East Asian girl with double braids sitting by the boat. The girl wears a white square collar puff sleeve dress, decorated with pleats and buttons. She has fair skin, delicate features, and slightly melancholic eyes, staring directly at the camera. Her hair falls naturally, with bangs covering part of her forehead. She rests her hands on the boat, appearing natural and relaxed. The background features a blurred outdoor scene, with hints of blue sky, mountains, and some dry plants. The video has a vintage film texture. A medium shot of a seated portrait.
2.  An anime illustration in vibrant thick painting style of a white girl with cat ears holding a folder, showing a slightly dissatisfied expression. She has long dark purple hair and red eyes, wearing a dark gray skirt and a light gray top with a white waist tie and a name tag in bold Chinese characters that says "Ziyang". The background has a light yellow indoor tone, with faint outlines of some furniture visible. A pink halo hovers above her head, in a smooth Japanese cel-shading style. A close-up shot from a slightly elevated perspective.
3.  CG game concept digital art featuring a huge crocodile with its mouth wide open, with trees and thorns growing on its back. The crocodile's skin is rough and grayish-white, resembling stone or wood texture. Its back is lush with trees, shrubs, and thorny protrusions. With its mouth agape, the crocodile reveals a pink tongue and sharp teeth. The background features a dusk sky with some distant trees, giving the overall scene a dark and cold atmosphere. A close-up from a low angle.
4.  In the style of an American drama promotional poster, Walter White sits in a metal folding chair wearing a yellow protective suit, with the words "Breaking Bad" written in sans-serif English above him, surrounded by piles of dollar bills and blue plastic storage boxes. He wears glasses, staring forward, dressed in a yellow jumpsuit, with his hands resting on his knees, exuding a calm and confident demeanor. The background shows an abandoned, dim factory with light filtering through the windows. There's a noticeable grainy texture. A medium shot with a straight-on close-up of the character.

**Directly output the generated English prompt.**
"""

# ... (The model loading and helper functions remain the same) ...
MODEL = None
PROCESSOR = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_qwen_model():
    global MODEL, PROCESSOR
    if MODEL is not None: return
    model_id = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/shared/hf-models/qwen/Qwen2.5-VL-32B-Instruct"
    print(f"Loading Qwen2.5-VL model '{model_id}' to {DEVICE}...")
    try:
        MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
        PROCESSOR = AutoProcessor.from_pretrained(model_id)
        print("Qwen2.5-VL model loaded successfully.")
    except Exception as e:
        sys.exit(f"Failed to load Qwen model: {e}")

def get_video_fps(video_path: Path) -> float:
    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps if fps > 0 else 2.0
    except Exception: return 2.0

def generate_video_description(video_path: Path) -> str | None:
    if not video_path.exists():
        tqdm.write(f"  [Warning] Video file not found: {video_path}")
        return None
    original_fps = get_video_fps(video_path)
    
    # <<< CHANGE: Use str(video_path) instead of video_path.as_uri()
    # This passes a clean, absolute filesystem path as a string, avoiding
    # the URI encoding issue with characters like '+'.
    messages = [{"role": "user", "content": [
        {"type": "video", "video": str(video_path), "fps": original_fps},
        {"type": "text", "text": VIDEO_DESCRIPTION_PROMPT},
    ]}]
    
    try:
        text = PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = PROCESSOR(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt", **video_kwargs).to(MODEL.device)
        generated_ids = MODEL.generate(**inputs, max_new_tokens=512, do_sample=False)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response_text = PROCESSOR.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response_text.strip()
    except Exception as e:
        tqdm.write(f"  [Error] Failed to generate description for {video_path.name}: {e}")
        return None

def save_json(data, path):
    """Helper function to save data to a JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Step 1: Generate video descriptions with recovery logic.")
    parser.add_argument("--json-file", type=Path, required=True, help="Path to the initial input JSON file.")
    parser.add_argument("--videos-dir", type=Path, required=True, help="Base directory of the videos.")
    parser.add_argument("--output-json", type=Path, required=True, help="Path to the working/output JSON file.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of *new* items to process.")
    parser.add_argument("--save-interval", type=int, default=10, help="Save progress every N items.")
    args = parser.parse_args()

    if args.output_json.exists():
        print(f"Resuming from existing file: {args.output_json}")
        with open(args.output_json, 'r') as f:
            working_data = json.load(f)
    else:
        print(f"Output file not found. Creating new working list from {args.json_file}.")
        with open(args.json_file, 'r') as f:
            all_videos = json.load(f)
        working_data = [
            item for item in all_videos
            if (response_value := item.get("response")) is not None  # 确保response不是None (处理JSON的null和缺失的键)
            and isinstance(response_value, str)                      # 确保response是字符串类型
            and response_value.strip().lower() != "none"             # 原始的字符串"none"检查
        ]
        save_json(working_data, args.output_json)

    items_to_process = [item for item in working_data if 'description_prompt' not in item]
    
    if not items_to_process:
        print("All descriptions have already been generated. Nothing to do.")
        sys.exit(0)
    
    if args.limit:
        print(f"Found {len(items_to_process)} items needing descriptions. Limiting to {args.limit}.")
        items_to_process = items_to_process[:args.limit]

    print(f"Processing {len(items_to_process)} videos to generate descriptions...")

    if DEVICE == "cpu": print("Warning: No CUDA device found. Script will be extremely slow.")
    load_qwen_model()

    for i, item in enumerate(tqdm(items_to_process, desc="Generating Descriptions")):
        input_video_path = args.videos_dir.resolve() / Path(item.get("video_path", ""))
        description = generate_video_description(input_video_path)
        if description:
            item['description_prompt'] = description
        
        if (i + 1) % args.save_interval == 0:
            tqdm.write(f"--- Saving progress ({i + 1}/{len(items_to_process)} processed) ---")
            save_json(working_data, args.output_json)

    print("\nFinal save of description data...")
    save_json(working_data, args.output_json)
    print("Step 1 complete.")

if __name__ == "__main__":
    main()