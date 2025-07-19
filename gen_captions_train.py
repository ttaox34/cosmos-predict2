import os
import torch
from pathlib import Path
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import decord
from tqdm import tqdm # 导入 tqdm 库

# --- CONFIGURATION ---
# Please modify these paths and the prompt before running the script.

# 1. Set the directory containing your .mp4 video files.
SOURCE_DIR = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/mingyu/cosmos-predict2/datasets/benchmark_train/custom_liquid/all_videos"

# 2. Set the directory where the .txt caption files will be saved.
#    The script will create this directory if it doesn't exist.
DEST_DIR = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/mingyu/cosmos-predict2/datasets/benchmark_train/custom_liquid/metas"

# 3. Choose the Qwen2.5-VL model to use. 
#    - "Qwen/Qwen2.5-VL-7B-Instruct" is a good balance.
#    - Other options: "Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-32B-Instruct", etc.
MODEL_ID = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/shared/hf-models/qwen/Qwen2.5-VL-32B-Instruct"

# 4. <<< IMPORTANT >>>
#    Hardcode your long, custom prompt here. This text will be used to generate
#    the caption for every video.
CUSTOM_PROMPT = """
You are a professional video scene analyst and creative prompt writer. Your goal is to watch the user's input video and generate a high-quality, detailed English prompt that accurately and vividly describes the scene. This generated prompt will be used to guide a video generation model.

Your output should strictly be the generated English prompt and nothing else. Do not include any greetings, explanations, or conversational text.

Task Requirements:

1.  Your primary task is to meticulously analyze the provided video and generate a comprehensive and visually rich description based solely on the visual information.
2.  Describe in detail the characteristics of the main subject(s), including their appearance, actions, expression, quantity, ethnicity, and posture. Also, describe the surrounding environment, overall style, spatial relationships, and camera angles.
3.  The entire output must be in English. If there is any text visible in the video (e.g., on signs, clothing, or book titles), retain it in its original form within quotes.
4.  You must carefully analyze the visual style of the video frames (e.g., cinematic, anime, photorealistic, documentary, vintage film) and incorporate a clear style description into the prompt.
5.  Describe any visible or implied movement. Even if the subject is static, describe their posture in a way that suggests potential action or the state they are in.
6.  Your output should convey natural movement attributes. Use simple and direct verbs to describe actions or potential actions related to the subject.
7.  Your description must be grounded in the visual evidence from the video frames. Pay close attention to details such as character clothing, accessories, background elements, lighting, and textures.
8.  Control the generated prompt to be around 80-100 words to ensure it is concise yet descriptive.

Example of the generated English prompt:

1.  A Japanese fresh film-style video of a young East Asian girl with double braids sitting by the boat. The girl wears a white square collar puff sleeve dress, decorated with pleats and buttons. She has fair skin, delicate features, and slightly melancholic eyes, staring directly at the camera. Her hair falls naturally, with bangs covering part of her forehead. She rests her hands on the boat, appearing natural and relaxed. The background features a blurred outdoor scene, with hints of blue sky, mountains, and some dry plants. The video has a vintage film texture. A medium shot of a seated portrait.
2.  An anime illustration in vibrant thick painting style of a white girl with cat ears holding a folder, showing a slightly dissatisfied expression. She has long dark purple hair and red eyes, wearing a dark gray skirt and a light gray top with a white waist tie and a name tag in bold Chinese characters that says "Ziyang". The background has a light yellow indoor tone, with faint outlines of some furniture visible. A pink halo hovers above her head, in a smooth Japanese cel-shading style. A close-up shot from a slightly elevated perspective.
3.  CG game concept digital art featuring a huge crocodile with its mouth wide open, with trees and thorns growing on its back. The crocodile's skin is rough and grayish-white, resembling stone or wood texture. Its back is lush with trees, shrubs, and thorny protrusions. With its mouth agape, the crocodile reveals a pink tongue and sharp teeth. The background features a dusk sky with some distant trees, giving the overall scene a dark and cold atmosphere. A close-up from a low angle.

**Output ONLY the generated English prompt. Do not include any other text.**
"""

# 5. Advanced settings for generation.
MAX_NEW_TOKENS = 2048  # Adjust based on the expected length of your captions.
# --- END OF CONFIGURATION ---


def main():
    """
    Main function to process videos and generate captions.
    """
    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Ensure source directory exists
    source_path = Path(SOURCE_DIR)
    if not source_path.is_dir():
        print(f"Error: Source directory not found at '{SOURCE_DIR}'")
        return

    # Create destination directory
    dest_path = Path(DEST_DIR)
    dest_path.mkdir(parents=True, exist_ok=True)
    print(f"Captions will be saved to: '{dest_path.resolve()}'")

    print(f"Loading model: {MODEL_ID}. This may take a while...")
    
    # Load the model and processor from Hugging Face based on the documentation
    # We use bfloat16 for better performance on modern GPUs and device_map="auto"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # For a potential speed-up, uncomment the line below if you have flash-attn installed
        # and your hardware supports it.
        # attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("Model and processor loaded successfully.")

    # Find all .mp4 files in the source directory
    video_files = list(source_path.glob("*.mp4"))
    print(f"Found {len(video_files)} video(s) to process.")

    # 关键修改：将 tqdm 实例赋值给一个变量 (e.g., pbar)
    pbar = tqdm(video_files, desc="Processing videos", unit="video")
    for video_file in pbar: # 迭代这个 pbar 对象
        output_txt_file = dest_path / video_file.with_suffix('.txt').name
        
        # Skip if the caption file already exists
        if output_txt_file.exists():
            # 使用 pbar.write 来避免干扰进度条
            pbar.write(f"Skipping '{video_file.name}', caption already exists.")
            continue

        # 关键修改：在 pbar 实例上调用 set_postfix
        pbar.set_postfix(file=video_file.name)
        # 移除原有的 "Processing 'video_file.name'..." 打印，因为 set_postfix 已经显示了

        try:
            # Step 1: Get the original frame rate of the video using decord
            video_reader = decord.VideoReader(str(video_file))
            original_fps = video_reader.get_avg_fps()
            # pbar.write(f"Detected original FPS: {original_fps:.2f}") # 如果这条信息不重要，可以注释掉

            # Step 2: Prepare the input messages for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": str(video_file),
                            "fps": original_fps / 3, 
                            "max_pixels": 640*360,
                        },
                        {"type": "text", "text": CUSTOM_PROMPT},
                    ],
                }
            ]

            # Step 3: Process the inputs using the processor
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            ).to(device)

            # Step 4: Generate the response from the model
            # pbar.write("Generating caption...") # 这条信息在循环中频繁出现可能会有点刷屏
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False  # Use greedy decoding for deterministic output
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            # Step 5: Save the generated caption to a text file
            with open(output_txt_file, "w", encoding="utf-8") as f:
                f.write(output_text)

            # 关键修改：在 pbar 实例上调用 write
            pbar.write(f"Successfully generated and saved caption to '{output_txt_file.name}'")
            # print("-" * 20 + " Caption " + "-" * 20)
            # print(output_text)
            # print("-" * 50)

        except Exception as e:
            # 关键修改：在 pbar 实例上调用 write
            pbar.write(f"An error occurred while processing '{video_file.name}': {e}")
            # Optionally, create an error log file
            error_log_file = output_txt_file.with_suffix('.error.log')
            with open(error_log_file, "w", encoding="utf-8") as f:
                f.write(str(e))

    print("\nAll videos have been processed.")


if __name__ == "__main__":
    main()