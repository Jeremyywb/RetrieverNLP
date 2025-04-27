import gzip
import json
import random
from pathlib import Path


def extract_examples(
    gz_path: str,
    output_path: str,
    num_examples: int = 2
) -> None:
    """
    从gzip压缩的JSON打包文件中随机抽取指定数量的示例并保存到新的JSON文件。

    参数:
        gz_path: 压缩文件路径，如 'json_pack_full_20250423_032758.json.gz'
        output_path: 输出文件路径，如 'input_examples.json'
        num_examples: 要选择的示例数量，默认为2
    """
    # 打开gz文件并按行读取JSON对象
    examples = []
    with gzip.open(gz_path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                examples.append(obj)
            except json.JSONDecodeError:
                continue

    if len(examples) < num_examples:
        raise ValueError(f"文件中的示例少于 {num_examples} 个，只有 {len(examples)} 个可用。")

    # 随机抽样或取前几个
    selected = random.sample(examples, num_examples)

    # 保存为新的JSON文件，每行一个对象
    output_file = Path(output_path)
    with output_file.open('w', encoding='utf-8') as f_out:
        for ex in selected:
            f_out.write(json.dumps(ex, ensure_ascii=False) + '\n')

    print(f"已保存 {num_examples} 个示例到 {output_path}")


if __name__ == '__main__':
    # 示例用法
    extract_examples(
        gz_path='json_pack_full_20250423_032758.json.gz',
        output_path='input_examples.json',
        num_examples=2
    )