# -*- coding: utf-8 -*-
# weibifan 2022-10-4
# PaddleHub，百度主导的PLM模型库，类似于HuggingFace Hub，允许第3方的PLMs
# 主要完成PaddleHub上模型的下载和使用，不包括微调。

"""  在2021年2月更新过一次，再无更新。大部分模型无法使用
预训练模型库：https://www.paddlepaddle.org.cn/hub

第1种方式：命令行方式

第2种方法：Python代码方式

第3种方式：Web 服务方式。
服务器端：PaddleHub Serving
客户端：使用Python代码

安装位置：
1）代码：C:\Users\Wei\.paddlehub
2）预训练的参数： C:\Users\Wei\.paddlenlp

打开Python Console 或 使用Python代码检查网络连接情况
import paddlehub
paddlehub.server_check()

中文句法解析  ddparser 无法使用

"""



import paddlehub as hub

'''
https://www.paddlepaddle.org.cn/hubdetail?name=lac
Lexical Analysis of Chinese，简称 LAC，是一个联合的词法分析模型，能整体性地完成中文分词、词性标注、专名识别任务。在百度自建数据集上评测，LAC效果：Precision=88.0%，Recall
=88.7%，F1-Score=88.4%。该PaddleHub Module支持预测。 

'''
lac = hub.Module(name="lac")
test_text = ["今天是个好天气。"]

results = lac.cut(text=test_text, use_gpu=False, batch_size=1, return_tag=True)
print(results)
#{'word': ['今天', '是', '个', '好天气', '。'], 'tag': ['TIME', 'v', 'q', 'n', 'w']}