# PunctuationPrediction-BERT
Use bert to predict punctuation on IWSLT2012 and The People's Daily 2014.

Data preprocessing can refer to this project: https://github.com/IsaacChanghau/neural_sequence_labeling

people_daily dataset: https://pan.baidu.com/s/1zTqAJJQ5OhGGPG6RQAa8AQ  pwd：bkc4


@20200916
Fork From https://github.com/MenNianShi/PunctuationPrediction-BERT.git

pip3 install tensorflow==1.9.0

glove.840B.300d 英语

data:
    datset/data_dir
    model(name)_dir
    raw:
        dataset

### 切分数据集
\data\raw\LREC 将 2014_corpus.txt 分 成 了 2014_train.txt 2014_dev.txt 2014_test.txt 三 部 分
dataprocess_peopledaily
split_dataset()


### 数据预处理
\data\dataset\lrec 对 2014_train.txt 2014_dev.txt 处理 形成 pd_train.json 和 pd_dev.json 文件
dataprocess_peopledaily
process_data(config)

### 训练
