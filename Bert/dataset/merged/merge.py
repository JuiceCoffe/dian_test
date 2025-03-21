import os
import shutil
import tempfile

def merge_jsonl_files(input_paths, output_path):
    """
    合并多个 JSONL 文件并生成包含总条目数的文件

    :param input_paths: list[str], 要合并的 JSONL 文件路径列表
    :param output_path: str, 输出文件路径（自动添加条目数后缀）
    :return: str, 最终生成的合并文件路径
    """
    # 创建临时文件
    temp_dir = os.path.dirname(output_path) or '.'  # 获取输出目录
    with tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', dir=temp_dir, delete=False
    ) as outfile:
        temp_name = outfile.name
        total_entries = 0
        
        try:
            for path in input_paths:
                # 验证输入文件是否存在
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"输入文件 {path} 不存在")
                
                # 读取并写入文件内容
                with open(path, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        outfile.write(line)
                        total_entries += 1
        except:
            # 发生异常时清理临时文件
            os.unlink(temp_name)
            raise

    # 生成最终文件名
    dir_path = os.path.dirname(output_path)
    base_name = os.path.basename(output_path)
    filename, ext = os.path.splitext(base_name)
    new_filename = f"{filename}-{total_entries}{ext}"
    new_output_path = os.path.join(dir_path, new_filename)

    # 确保输出目录存在
    os.makedirs(dir_path, exist_ok=True)

    # 移动临时文件到最终路径
    shutil.move(temp_name, new_output_path)

    return new_output_path

# 示例用法
if __name__ == "__main__":
    # 输入文件列表（替换为实际路径）
    input_files = [
        "/data/haominpeng/dataset_score/used/comments_and_ratings1_1.jsonl",
        "/data/haominpeng/dataset_score/used/comments_and_ratings50_53.jsonl",
        "/data/haominpeng/dataset_score/used/comments_and_ratings80_89.jsonl",
        "/data/haominpeng/dataset_score/used/comments_and_ratings240_249.jsonl",
        "/data/haominpeng/dataset_score/used/comments_and_ratings270_279.jsonl",
        "/data/haominpeng/dataset_score/used/comments_and_ratings340_349.jsonl",
        "/data/haominpeng/dataset_score/used/comments_and_ratings350_359.jsonl",
        "/data/haominpeng/dataset_score/used/comments_and_ratings360_369.jsonl",
        "/data/haominpeng/dataset_score/used/comments_and_ratings370_380.jsonl",
        "/data/haominpeng/dataset_score/used/comments_and_ratings320_329.jsonl"
    ]
    
    # 输出路径（文件名部分会自动添加条目数）
    output = "/data/haominpeng/dataset_score/merged/combined.jsonl"
    
    # 执行合并
    result_path = merge_jsonl_files(input_files, output)
    print(f"合并完成，文件已保存至：{result_path}")