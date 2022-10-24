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

MODEL_NAME = "ernie-3.0-medium-zh"
tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained(MODEL_NAME)
ernie_model = ppnlp.transformers.ErnieModel.from_pretrained(MODEL_NAME)

# 将原始输入文本切分token，
#tokens = tokenizer._tokenize("请输入测试样例。This a test case.")
# 一行代码完成切分token，映射token ID以及拼接特殊token
encoded_text = tokenizer(text="请输入测试样例")
for key, value in encoded_text.items():
    print("{}:\n\t{}".format(key, value))

# 转化成paddle框架数据格式
input_ids = paddle.to_tensor([encoded_text['input_ids']])
print("input_ids : {}".format(input_ids))
segment_ids = paddle.to_tensor([encoded_text['token_type_ids']])
print("token_type_ids : {}".format(segment_ids))

# 此时即可输入ERNIE模型中得到相应输出
sequence_output, pooled_output = ernie_model(input_ids, segment_ids)
print("Token wise output: {}, Pooled output: {}".format(sequence_output.shape, pooled_output.shape))

'''
input_ids:
	[1, 647, 789, 109, 558, 525, 314, 656, 2]
token_type_ids:
	[0, 0, 0, 0, 0, 0, 0, 0, 0]
input_ids : Tensor(shape=[1, 9], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
       [[1  , 647, 789, 109, 558, 525, 314, 656, 2  ]])
token_type_ids : Tensor(shape=[1, 9], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
       [[0, 0, 0, 0, 0, 0, 0, 0, 0]])
Token wise output: [1, 9, 768], Pooled output: [1, 768]
'''

# 单句输入
single_seg_input = tokenizer(text="请输入测试样例")
# 句对输入
multi_seg_input = tokenizer(text="请输入测试样例1", text_pair="请输入测试样例2")

print("单句输入token (str): {}".format(tokenizer.convert_ids_to_tokens(single_seg_input['input_ids'])))
print("单句输入token (int): {}".format(single_seg_input['input_ids']))
print("单句输入segment ids : {}".format(single_seg_input['token_type_ids']))

print()
print("句对输入token (str): {}".format(tokenizer.convert_ids_to_tokens(multi_seg_input['input_ids'])))
print("句对输入token (int): {}".format(multi_seg_input['input_ids']))
print("句对输入segment ids : {}".format(multi_seg_input['token_type_ids']))

'''
单句输入token (str): ['[CLS]', '请', '输', '入', '测', '试', '样', '例', '[SEP]']
单句输入token (int): [1, 647, 789, 109, 558, 525, 314, 656, 2]
单句输入segment ids : [0, 0, 0, 0, 0, 0, 0, 0, 0]

句对输入token (str): ['[CLS]', '请', '输', '入', '测', '试', '样', '例', '1', '[SEP]', '请', '输', '入', '测', '试', '样', '例', '2', '[SEP]']
句对输入token (int): [1, 647, 789, 109, 558, 525, 314, 656, 208, 2, 647, 789, 109, 558, 525, 314, 656, 249, 2]
句对输入segment ids : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
'''

# Highlight: padding到统一长度
encoded_text = tokenizer(text="请输入测试样例",  max_seq_len=15)

for key, value in encoded_text.items():
    print("{}:\n\t{}".format(key, value))