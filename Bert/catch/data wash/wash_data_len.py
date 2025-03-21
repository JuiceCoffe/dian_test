import json
import re
import jieba
from zhon.hanzi import punctuation
from tqdm import tqdm
from collections import defaultdict

def clean_text(text):
    """
    深度清洗中文文本
    （原有清洗逻辑保持不变）
    """
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 去除URL
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\$\$,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # 全角转半角
    text = text.translate(str.maketrans(
        'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ０１２３４５６７８９',
        'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    ))
    
    # 处理特殊空白字符
    text = re.sub(r'[\r\n\t\v\f]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 过滤非常用符号
    allowed_symbols = punctuation + '!?,.~@#%&_=:;'
    text = re.sub(f'[^{re.escape(allowed_symbols)}\w\s]', '', text)
    
    # 去除重复标点
    text = re.sub(r'([!?,.])\1{2,}', r'\1', text)
    
    return text

def is_valid_sample(text, score, min_length):
    """
    综合验证样本有效性
    新增根据评分动态传入min_length参数
    """
    # 检查评分范围
    if not (1 <= score <= 10):
        return False, "invalid_score"
    
    # 去除空格后的文本长度检查
    clean_len = len(text.strip())
    if clean_len < min_length:
        return False, "too_short"
    
    # 有效中文字符比例检查
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    if len(chinese_chars) / (clean_len + 1e-6) < 0.5:
        return False, "low_chinese_ratio"
    
    # 重复内容检查
    words = list(jieba.cut(text))
    unique_words = set(words)
    if len(unique_words) / len(words) < 0.3:
        return False, "high_repetition"
    
    return True, "valid"

def data_cleaning(input_path, output_path, score_min_len_dict):
    """
    主清洗函数
    新增评分长度字典参数和类别分布统计
    """
    # 验证字典完整性
    for score in range(1, 11):
        if score not in score_min_len_dict:
            raise ValueError(f"参数字典缺少评分 {score} 的min_len定义")

    stats = defaultdict(int)
    category_distribution = defaultdict(int)
    cleaned_data = []
    seen_texts = set()

    with open(input_path, 'r', encoding='utf-8') as f_in:
        for line in tqdm(f_in, desc='Processing'):
            try:
                data = json.loads(line)
                raw_text = data['text']
                raw_score = int(data['point'])
                
                text = clean_text(raw_text)
                min_length = score_min_len_dict[raw_score]
                validation, reason = is_valid_sample(text, raw_score, min_length)
                
                # 检查重复内容
                text_hash = hash(text.strip().lower())
                if text_hash in seen_texts:
                    stats['duplicate'] += 1
                    continue
                seen_texts.add(text_hash)
                
                if validation:
                    cleaned_data.append({
                        'text': text,
                        'point': raw_score
                    })
                    stats['valid'] += 1
                    category_distribution[raw_score] += 1
                else:
                    stats[reason] += 1
                
            except Exception as e:
                stats['parse_error'] += 1

    # 保存清洗结果
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for item in cleaned_data:
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 输出统计信息
    print("\n清洗统计:")
    total = sum(stats.values())
    print(f"原始数据总量: {total}")
    print(f"有效保留数据: {stats['valid']} ({stats['valid']/total:.1%})")
    
    print("\n无效数据分布:")
    for reason, count in stats.items():
        if reason not in ['valid', 'duplicate']:
            print(f"- {reason}: {count} ({count/total:.1%})")
    print(f"- duplicate: {stats['duplicate']} ({stats['duplicate']/total:.1%})")

    # 新增类别分布报告
    print("\n评分类别分布（有效数据）:")
    total_valid = stats['valid']
    for score in sorted(score_min_len_dict.keys()):
        count = category_distribution.get(score, 0)
        percentage = count / total_valid * 100 if total_valid > 0 else 0
        print(f"评分 {score}（min_len={score_min_len_dict[score]}）: {count} 条 ({percentage:.1f}%)")

    return cleaned_data

if __name__ == "__main__":
    # 示例评分参数配置（请根据实际需求修改）
    SCORE_MIN_LEN = {
        1: 12,  2: 12, 3: 14, 4: 37, 5: 39,
        6: 42, 7: 40, 8: 30, 9: 32, 10: 30
    }

    input_file = "/data/haominpeng/dataset_score/merged/combined-60049.jsonl"
    output_file = "/data/haominpeng/Work/dian/exam/Bert/catch/combined_60049_cleaned_len.jsonl"
    
    print("开始数据清洗...")
    cleaned = data_cleaning(input_file, output_file, SCORE_MIN_LEN)
    print(f"\n清洗完成，有效样本数：{len(cleaned)}")