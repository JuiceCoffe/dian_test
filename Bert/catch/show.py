import json
from collections import defaultdict

def analyze_jsonl(file_path):
    """
    分析JSONL文件并打印统计信息
    返回格式：
    - 类别数量分布
    - 类别平均文本长度
    - 总体统计数据
    """
    # 初始化统计数据结构
    category_stats = defaultdict(lambda: {'count': 0, 'total_length': 0})
    overall_stats = {'total_count': 0, 'total_length': 0}

    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                text = data.get('text', '')
                point = data.get('point', -1)  # 使用-1表示缺失类别
                
                # 更新统计
                text_len = len(text)
                category_stats[point]['count'] += 1
                category_stats[point]['total_length'] += text_len
                overall_stats['total_count'] += 1
                overall_stats['total_length'] += text_len
            except json.JSONDecodeError:
                print(f"警告：跳过无法解析的行: {line}")
                continue

    # 计算平均值
    for point in category_stats:
        stats = category_stats[point]
        stats['avg_length'] = stats['total_length'] / stats['count'] if stats['count'] else 0

    overall_avg = overall_stats['total_length'] / overall_stats['total_count'] if overall_stats['total_count'] else 0

    # 打印结果
    print("类别统计报告")
    print("=" * 40)
    print(f"{'类别':<8} {'数量':<8} {'平均长度':<10}")
    for point in sorted(category_stats.keys()):
        stats = category_stats[point]
        print(f"{point:<8} {stats['count']:<8} {stats['avg_length']:.2f}")
    
    print("\n总体统计")
    print("=" * 40)
    print(f"总样本数: {overall_stats['total_count']}")
    print(f"总体平均长度: {overall_avg:.2f} 字符")
    print(f"类别数: {len(category_stats)}")

if __name__ == "__main__":
    # 替换为实际文件路径
    input_file = "/data/haominpeng/Work/dian/exam/Bert/catch/combined_45457_cleaned_num.jsonl"  
    analyze_jsonl(input_file)