# Poetry-Pytorch

Char-level poetry generator with Transformer and LSTM based on Pytorch.

## requirements
Pytorch >=1.2

You can install the requirements by:
```pip install -r requirements.txt```

## Introduction
这是字符级的古诗生成器，支持三种模型：
1. Transformer
  包含encoder decoder结构，encoder对诗词标题编码，decoder生成诗词内容。
2. Transformer LM
  仅使用Transformer的encoder作为语言模型，对诗词进行生成。
3. LSTM
  使用LSTM作为语言模型，对诗词进行生成。

在使用1模型时，使用dataset/poetry.txt，其中包含标题与诗词。每一行是整首诗词，文本最大长度需要设置长一些。
```
python main.py --model transformer --hidden_dims 512 --tgt_max_len 64 --num_epochs 100 --batch_size 64 --lr 1e-3 --train --data_path "dataset/poetry.txt" --output_path "output/model-transformer.pkl"
```

使用模型2：
```
python main.py --model transformerlm --hidden_dims 512 --tgt_max_len 16 --num_epochs 100  --batch_size 64 --lr 1e-3 --train --data_path "dataset/poetry-no.txt" --output_path "output/model-transformerlm.pkl"
```

使用模型3：
```
python main.py --model lstm --hidden_dims 512 --tgt_max_len 16 --num_epochs 100 --batch_size 64 --lr 1e-3 --train --data_path "dataset/poetry-no.txt" --output_path "output/model-lstm.pkl"
```

测试：
使用模型1：
```
python main.py --model transformer --tgt_max_len 64 --test --data_path dataset/poetry.txt --output_path output/model-transformer.pkl --src_text "沁园春" --tgt_text ""
```
在使用模型2、3时不需要src_text，因为只有语言模型。

更多参数可以查看main.py中的Opt()函数，或运行```python main.py -h```查看。

测试时注意--model与所保存的模型要一致，--data_path在测试时也需要，要生成词表。

模型效果：
```
春风吹玉管，夜雨洒金羁。

梦人不可见，不觉白头吟。
```

## 文件结构说明
```
-data/
 -process.py 处理数据
-dataset/
 数据文件
-models/
 -model.py 模型
 -trainer.py 训练方法
-main.py 运行的入口文件