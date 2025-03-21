# 进度说明
各个任务提交在Bert、attention、random forest 3个文件夹里

# 项目目录
  dian_upload/  
  ├── Bert/  
  │   ├── bert_classify_test.py                               //用于直接推理  
  │   ├── bert_classify_train.py                          //分类任务，效果较好，最终采用  
  │   ├── bert_ordinary_regression_test.py.py  
  │   ├── bert_ordinary_regression_train.py               //序列回归任务，效果不如分类  
  │   ├── bert_regression_train.py                        //回归任务，效果较差  
  │   ├── catch/                                          //数据爬取                                     
  │   │   ├── catch_data.py  
  │   │   ├── catch_data_pro.py  
  │   │   ├── data wash/                                  //根据长度、数量进行清洗          
  │   │   │   ├── wash_data.py  
  │   │   │   ├── wash_data_len.py  
  │   │   │   ├── wash_data_num.py  
  │   │   ├── show.py  
  │   ├── loss_fn_test.py                                 //损失函数调试  
  ├── attention/                                          //几种注意力实现  
  │   ├── Multi_Head_Latent_Attention.py  
  │   ├── base_multi_attention.py  
  ├── random forest/                                      //随机森林实现  
  │   ├── Random_forest.py  
  │   ├── iris.csv  
  ├── README.md  
