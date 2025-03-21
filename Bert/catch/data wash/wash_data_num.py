import json
import re
import jieba
from zhon.hanzi import punctuation
from tqdm import tqdm
from collections import defaultdict

def clean_text(text):
    """
    深度清洗中文文本
    处理流程：
    1. 去除HTML标签
    2. 去除URL链接
    3. 统一全角/半角符号
    4. 处理特殊空白字符
    5. 过滤非常用符号
    6. 去除重复标点
    7. 基本繁体转简体（需安装opencc-python）
    """
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 去除URL
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # 全角转半角
    text = text.translate(str.maketrans(
        'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ０１２３４５６７８９',
        'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    ))
    
    # 处理特殊空白字符
    text = re.sub(r'[\r\n\t\v\f]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 过滤非常用符号（保留中文标点和常用符号）
    allowed_symbols = punctuation + '!?,.~@#%&_=:;'
    text = re.sub(f'[^{re.escape(allowed_symbols)}\w\s]', '', text)
    
    # 去除重复标点
    text = re.sub(r'([!?,.])\1{2,}', r'\1', text)
    
    return text

def is_valid_sample(text, score, min_length=0):
    """
    调整有效性检查，默认不限制长度
    """
    # 检查评分范围
    if not (1 <= score <= 10):
        return False, "invalid_score"
    
    # 去除空格后的文本长度检查（传入的min_length为0时不限制）
    clean_len = len(text.strip())
    if clean_len < min_length:
        return False, "too_short"
    
    # 有效中文字符比例检查
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    if len(chinese_chars) / (clean_len + 1e-6) < 0.5:
        return False, "low_chinese_ratio"
    
    # 重复内容检查（基于分词）
    words = list(jieba.cut(text))
    unique_words = set(words)
    if len(unique_words) / len(words) < 0.3:
        return False, "high_repetition"
    
    return True, "valid"

def data_cleaning(input_path, output_path=None):
    """
    返回清洗后的数据，可选保存到文件
    """
    stats = defaultdict(int)
    cleaned_data = []
    seen_texts = set()
    
    with open(input_path, 'r', encoding='utf-8') as f_in:
        for line in tqdm(f_in, desc='Processing'):
            try:
                data = json.loads(line)
                raw_text = data['text']
                raw_score = int(data['point'])
                
                text = clean_text(raw_text)
                validation, reason = is_valid_sample(text, raw_score)  # min_length默认为0
                
                text_hash = hash(text.strip().lower())
                if text_hash in seen_texts:
                    stats['duplicate'] += 1
                    continue
                seen_texts.add(text_hash)
                
                if validation:
                    cleaned_data.append({'text': text, 'point': raw_score})
                    stats['valid'] += 1
                else:
                    stats[reason] += 1
            except:
                stats['parse_error'] += 1
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for item in cleaned_data:
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("\n初步清洗统计:")
    print(f"总有效样本（不含长度限制）: {stats['valid']}")
    return cleaned_data

def process_per_num(cleaned_data, per_num):
    point_data = defaultdict(list)
    for item in cleaned_data:
        point = item['point']
        text_len = len(item['text'].strip())
        point_data[point].append((text_len, item))
    
    processed_data = []
    min_len_dict = {}
    insufficient_points = []
    
    for point in point_data:
        samples = point_data[point]
        sorted_samples = sorted(samples, key=lambda x: (-x[0], x[1]['text']))
        total = len(sorted_samples)
        
        if total < per_num:
            insufficient_points.append((point, total))
            selected = sorted_samples
        else:
            selected = sorted_samples[:per_num]
        
        min_len = selected[-1][0] if selected else 0
        min_len_dict[point] = min_len
        
        for s in selected:
            processed_data.append(s[1])
    
    # 输出结果
    if insufficient_points:
        print("\n以下类别数据不足:")
        for p, cnt in insufficient_points:
            print(f"类别 {p}: 现有 {cnt} 条, 期望 {per_num} 条")
        exit(0)
    
    print("\n各类别最小长度min_len:")
    for p in sorted(min_len_dict):
        print(f"类别 {p}: {min_len_dict[p]}")
    
    return processed_data

if __name__ == "__main__":

    per_num = 1080 # 用户定义的每个类别期望数量
    input_file = "/data/haominpeng/Work/dian/exam/Bert/catch/combined_60049_cleaned_len.jsonl"
    output_file = f"/data/haominpeng/Work/dian/exam/Bert/catch/combined_60049_pernum={1091}.jsonl"
    print("开始数据清洗...")
    cleaned = data_cleaning(input_file)
    
    print("\n按类别动态调整样本...")
    final_data = process_per_num(cleaned, per_num)
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in final_data:
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n处理完成，最终样本数：{len(final_data)}")