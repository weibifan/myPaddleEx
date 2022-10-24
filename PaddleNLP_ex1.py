# -*- coding: utf-8 -*-
# weibifan 2022-10-8
# PaddleNLP，中文自然语言处理的工具，可以完成PLMs的下载，微调，及使用
# https://www.paddlepaddle.org.cn/paddle/paddlenlp

'''
中文分词，中英文混合分词。   这里分词只是分字

预训练模型ERNIE对中文数据的处理是以字为单位。
'''

import paddle
import paddlenlp as ppnlp

# 目的：将一个句子表示成一个768维的向量。也就是sent2vec
MODEL_NAME = "ernie-3.0-medium-zh"

# 实际上是加载了和这个模型一起的另外一个vocab.txt字典文件
# 对于分类任务来说，可以用字建立特征。ernie无法用词
tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained(MODEL_NAME)

ernie_model = ppnlp.transformers.ErnieModel.from_pretrained(MODEL_NAME)

# 将原始输入文本切分token，
tokens = tokenizer._tokenize("请输入测试样例。This a test case.")
print("Tokens: {}".format(tokens))

# token映射为对应token id
tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
print("Tokens id: {}".format(tokens_ids))


# 拼接上预训练模型对应的特殊token ，如[CLS]、[SEP]
tokens_ids = tokenizer.build_inputs_with_special_tokens(tokens_ids)

# 转化成paddle框架数据格式
tokens_pd = paddle.to_tensor([tokens_ids])
print("Tokens : {}".format(tokens_pd))

# 此时即可输入ERNIE模型中得到相应输出，将一个句子表示成一个768维的向量。
sequence_output, pooled_output = ernie_model(tokens_pd)
print("Token wise output: {}, Pooled output: {}".format(sequence_output.shape, pooled_output.shape))

'''

Tokens: ['请', '输', '入', '测', '试', '样', '例', '。', 'this', 'a', 'test', 'case', '.']
Tokens id: [647, 789, 109, 558, 525, 314, 656, 12043, 3730, 1545, 6943, 7977, 42]
Tokens : Tensor(shape=[1, 15], dtype=int64, place=Place(cpu), stop_gradient=True,
       [[1   , 647 , 789 , 109 , 558 , 525 , 314 , 656 , 12043, 3730, 1545, 6943,
         7977, 42  , 2   ]])
Token wise output: [1, 15, 768], Pooled output: [1, 768]

'''