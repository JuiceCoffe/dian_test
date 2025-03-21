# 进度说明
基本完成所有任务，各个任务提交在Bert、attention、random forest 3个文件夹里

# 项目目录
dian_upload/
├── Bert/
│   ├── bert_classify_test.py                           //用于直接推理
│   ├── bert_classify_train.py                          //分类任务，效果较好，最终采用
│   ├── bert_ordinary_regression_test.py.py
│   ├── bert_ordinary_regression_train.py               //序列回归任务，效果不如分类
│   ├── bert_regression_train.py                        //回归任务，效果较差
│   ├── catch/                                          //数据爬取及最终采用的数据集                                    
│   │   ├── catch_data.py
│   │   ├── catch_data_pro.py
│   │   ├── combined_60049_cleaned_low.jsonl
│   │   ├── combined_60049_minlen=30.jsonl
│   │   ├── combined_60049_pernum=1500.jsonl
│   │   ├── data wash/                                  //根据长度、数量进行清洗        
│   │   │   ├── wash_data.py
│   │   │   ├── wash_data_len.py
│   │   │   ├── wash_data_num.py
│   │   ├── show.py
│   │   ├── test.jsonl
│   ├── dataset/                                        //爬取的原始数据
│   │   ├── merged/
│   │   │   ├── combined-60049.jsonl
│   │   │   ├── merge.py
│   │   ├── used/
│   │   │   ├── comments_and_ratings1_1.jsonl
│   │   │   ├── comments_and_ratings240_249.jsonl
│   │   │   ├── comments_and_ratings270_279.jsonl
│   │   │   ├── comments_and_ratings320_329.jsonl
│   │   │   ├── comments_and_ratings340_349.jsonl
│   │   │   ├── comments_and_ratings350_359.jsonl
│   │   │   ├── comments_and_ratings360_369.jsonl
│   │   │   ├── comments_and_ratings370_380.jsonl
│   │   │   ├── comments_and_ratings50_53.jsonl
│   │   │   ├── comments_and_ratings80_89.jsonl
│   ├── loss_fn_test.py
├── attention/                                          //几种注意力实现
│   ├── Multi_Head_Latent_Attention.py
│   ├── base_multi_attention.py
├── random forest/                                      //随机森林实现
│   ├── Random_forest.py
│   ├── iris.csv
├── README.md