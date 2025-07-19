import json
import os

def transform_json_file(input_filepath, output_filepath):
    """
    转换JSON文件：
    1. 移除除video_path和response外的所有键。
    2. video_path键改名为raw_video_path，并修改其值的前缀。
    3. 新增gen_video_path键，根据raw_video_path的文件名和编号生成新路径。

    Args:
        input_filepath (str): 输入JSON文件的路径。
        output_filepath (str): 输出JSON文件的路径。
    """
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：文件未找到 '{input_filepath}'")
        return
    except json.JSONDecodeError:
        print(f"错误：无法解析JSON文件 '{input_filepath}'。请检查文件格式。")
        return

    transformed_data = []
    prefix_to_remove = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/mingyu/datasets/"
    new_raw_prefix = "raw_videos/"
    gen_video_base_path = "gen_videos/GRASP/cosmos2/triggers/"

    for i, item in enumerate(data):
        new_item = {}

        # 1. 处理 raw_video_path
        original_video_path = item.get("video_path", "")
        if original_video_path:
            # 替换前缀
            modified_raw_video_path = original_video_path.replace(prefix_to_remove, new_raw_prefix)
            new_item["raw_video_path"] = modified_raw_video_path
        else:
            new_item["raw_video_path"] = "" # 如果原路径为空，则也设置为空

        # 2. 保留 response
        new_item["response"] = item.get("response", "")

        # 3. 生成 gen_video_path
        if original_video_path:
            # 获取文件名（例如 "10kuo2q.mp4"）
            filename = os.path.basename(original_video_path)
            # 分离文件名和扩展名（例如 "10kuo2q", ".mp4"）
            name_part, ext_part = os.path.splitext(filename)
            # 构建带编号的文件名（例如 "10kuo2q_1.mp4"）
            gen_filename = f"{name_part}_{i + 1}{ext_part}" # i+1 因为编号从1开始
            # 组合完整路径
            full_gen_video_path = os.path.join(gen_video_base_path, gen_filename).replace("\\", "/") # 确保路径使用正斜杠
            new_item["gen_video_path"] = full_gen_video_path
        else:
            new_item["gen_video_path"] = "" # 如果原路径为空，则也设置为空

        transformed_data.append(new_item)

    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(transformed_data, f, indent=2, ensure_ascii=False)
        print(f"转换成功！输出文件已保存到 '{output_filepath}'")
    except IOError:
        print(f"错误：无法写入输出文件 '{output_filepath}'。")

# --- 使用示例 ---
if __name__ == "__main__":
    input_json_file = "GRASP.json"  # 你的输入文件路径
    output_json_file = "GRASP_processed.json" # 你希望保存的输出文件路径

    # 执行转换
    transform_json_file(input_json_file, output_json_file)

    # 验证输出文件内容（可选）
    print("\n--- 输出文件内容预览 ---")
    try:
        with open(output_json_file, 'r', encoding='utf-8') as f:
            output_content = json.load(f)
            # 打印前几条或全部
            print(json.dumps(output_content, indent=2, ensure_ascii=False))
    except FileNotFoundError:
        pass # 错误信息已在transform_json_file中处理