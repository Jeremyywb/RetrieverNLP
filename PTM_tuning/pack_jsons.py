import os
import gzip
import json
import argparse
from datetime import datetime

def pack_json_to_gz(directory: str, output_dir: str):
    name_input = directory
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    sub_dir = 'analysis'
    directory = os.path.join(ROOT_DIR, sub_dir, directory)
    
    # 输出文件名：自动命名为 json_pack_时间戳.json.gz
    filename = f"json_pack_{name_input}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json.gz"
    output_gz_path = os.path.join(ROOT_DIR, output_dir, filename)

    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(output_gz_path), exist_ok=True)

    print(f"保存位置：{output_gz_path}")

    with gzip.open(output_gz_path, 'wt', encoding='utf-8') as gz_file:
        file_count = 0
        for fname in os.listdir(directory):
            if fname.endswith('.json'):
                fpath = os.path.join(directory, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                        gz_file.write(json.dumps(content, ensure_ascii=False) + '\n')
                        file_count += 1
                except Exception as e:
                    print(f"处理文件 {fname} 时出错: {e}")
        print(f"✅ 共打包 {file_count} 个 JSON 文件。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将指定目录下的所有 .json 文件打包到指定输出目录中")
    parser.add_argument('--directory', type=str, required=True, help='analysis 下的子目录名')
    parser.add_argument('--outputgz', type=str, required=True, help='输出目录名，例如 input（会保存到 ../PTM_tuning/input 下）')
    args = parser.parse_args()

    pack_json_to_gz(args.directory, args.outputgz)
