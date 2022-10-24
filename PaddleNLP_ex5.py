# -*- coding: utf-8 -*-
# weibifan 2022-10-9
# PaddleNLP一键预测功能：Taskflow API
# https://paddlenlp.readthedocs.io/zh/latest/model_zoo/taskflow.html



'''
https://aistudio.baidu.com/aistudio/projectdetail/3696243

基本任务包括：中文分词，信息抽取，词性标注，命名实体识别，依存句法分析。
行业任务：中文纠错，情感分类，文本相似度，知识标注
其他：交互式闲聊，生成式问答，智能写诗。

'''
from paddlenlp import Taskflow

# 中文文本纠错
'''
corrector = Taskflow("text_correction")
print(corrector('遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇。'))
'''


# 使用skep_ernie_1.0_large_ch模型进行情感分析

senta = Taskflow("sentiment_analysis", model="skep_ernie_1.0_large_ch") # model不支持ernie-3.0-medium-zh
senta("作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。")

