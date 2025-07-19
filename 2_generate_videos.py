# Save this as 2_generate_videos.py
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm

def save_json(data, path):
    """Helper function to save data to a JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def run_video_generation(input_video_path: Path, output_dir: Path, prompt: str, process_index: int) -> Path | None:
    original_stem, original_suffix = input_video_path.stem, input_video_path.suffix
    new_filename = f"{original_stem}_{process_index}{original_suffix}"
    output_video_path = output_dir / new_filename
    
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "torchrun", "--nproc_per_node=4", "examples/video2world.py",
        "--model_size", "2B", "--input_path", str(input_video_path),
        "--prompt", prompt, "--save_path", str(output_video_path),
        "--num_gpus", "4", "--num_conditional_frames", "5",
        "--disable_guardrail", "--disable_prompt_refiner",
    ]

    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = f".{os.pathsep}{child_env.get('PYTHONPATH', '')}"
    
    try:
        subprocess.run(command, check=True, env=child_env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return output_video_path
    except subprocess.CalledProcessError as e:
        tqdm.write(f"\n[Error] Video generation failed for {input_video_path.name}.")
        if e.stderr: tqdm.write(f"--- Subprocess Error Output ---\n{e.stderr.decode('utf-8')}\n-----------------------------")
        return None
    except Exception as e:
        tqdm.write(f"\n[Error] An unexpected error occurred during video generation for {input_video_path.name}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Step 2: Generate videos with recovery logic.")
    parser.add_argument("--work-json", type=Path, required=True, help="Path to the JSON file with prompts (will be updated in-place).")
    parser.add_argument("--videos-dir", type=Path, required=True, help="Base directory of the original videos.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save generated videos.")
    parser.add_argument("--save-interval", type=int, default=5, help="Save progress every N items.")
    args = parser.parse_args()
    
    if not args.work_json.exists():
        sys.exit(f"Error: Working JSON file not found at {args.work_json}. Please run Step 1 first.")

    with open(args.work_json, 'r') as f:
        working_data = json.load(f)

    # <<< CHANGE: Filter for items that need video generation
    items_to_process = [item for item in working_data if 'generated_video_path' not in item and 'description_prompt' in item]

    if not items_to_process:
        print("All videos have already been generated. Nothing to do.")
        sys.exit(0)
    
    print(f"Found {len(items_to_process)} videos to generate.")

    for i, item in enumerate(tqdm(items_to_process, desc="Generating Videos")):
        prompt = item.get('description_prompt')
        input_video_path = args.videos_dir.resolve() / Path(item.get("video_path", ""))
        
        # Get the original index to create a consistent filename across runs
        original_index = working_data.index(item) + 1
        
        generated_path = run_video_generation(input_video_path, args.output_dir.resolve(), prompt, process_index=original_index)
        
        if generated_path:
            item['generated_video_path'] = str(generated_path)
        else:
            # Optionally mark as failed to avoid retrying a consistently failing item
            item['generated_video_path'] = "FAILED"

        if (i + 1) % args.save_interval == 0:
            tqdm.write(f"--- Saving progress ({i + 1}/{len(items_to_process)} processed) ---")
            save_json(working_data, args.work_json)

    print("\nFinal save of video generation data...")
    save_json(working_data, args.work_json)
    print("Step 2 complete.")

if __name__ == "__main__":
    main()