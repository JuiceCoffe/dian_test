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

def is_valid_sample(text, score, min_length):
    """
    综合验证样本有效性
    返回有效性状态和原因
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
    
    # 重复内容检查（基于分词）
    words = list(jieba.cut(text))
    unique_words = set(words)
    if len(unique_words) / len(words) < 0.3:
        return False, "high_repetition"
    
    return True, "valid"

def data_cleaning(input_path, output_path,min_length):
    """
    主清洗函数
    返回清洗统计信息
    """
    stats = defaultdict(int)
    cleaned_data = []
    seen_texts = set()  # 用于检测重复内容
    
    with open(input_path, 'r', encoding='utf-8') as f_in:
        # 使用tqdm显示进度条
        for line in tqdm(f_in, desc='Processing'):
            try:
                data = json.loads(line)
                raw_text = data['text']
                raw_score = int(data['point'])
                
                # 执行清洗流程
                text = clean_text(raw_text)
                validation, reason = is_valid_sample(text, raw_score,min_length)
                
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
                else:
                    stats[reason] += 1
                
            except Exception as e:
                stats['parse_error'] += 1
    
    # 保存清洗后的数据
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for item in cleaned_data:
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 打印统计信息
    print("\n清洗统计:")
    print(f"原始数据总量: {sum(stats.values())}")
    print(f"有效保留数据: {stats['valid']} ({stats['valid']/sum(stats.values()):.1%})")
    print("无效数据分布:")
    for k, v in stats.items():
        if k != 'valid':
            print(f"- {k}: {v} ({v/sum(stats.values()):.1%})")
    
    return cleaned_data

if __name__ == "__main__":
    MIN_LEN=30
    input_file = "/data/haominpeng/dataset_score/merged/combined-60049.jsonl"
    output_file = f"/data/haominpeng/Work/dian/exam/Bert/catch/combined_60049_minlen={MIN_LEN}.jsonl"
    print("开始数据清洗...")
    cleaned = data_cleaning(input_file, output_file,MIN_LEN)
    print(f"\n清洗完成，有效样本数：{len(cleaned)}")