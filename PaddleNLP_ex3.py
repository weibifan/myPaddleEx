# -*- coding: utf-8 -*-
# weibifan 2022-10-9
# PaddleNLP Transformer模型的使用：PaddleNLP Transformer API
# https://paddlenlp.readthedocs.io/zh/latest/model_zoo/index.html

'''

模型总数：大约30多个，其中有中文处理模型
PaddleNLP的Transformer预训练模型包含从 huggingface.co 直接转换的模型权重和百度自研模型权重，
方便社区用户直接迁移使用。 目前共包含了40多个主流预训练模型，500多个模型权重。

4类任务：序列任务，token任务，QA任务，文本生成任务。

'''

from functools import partial
import numpy as np

import paddle
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer

train_ds = load_dataset("chnsenticorp", splits=["train"])

model = AutoModelForSequenceClassification.from_pretrained("bert-wwm-chinese", num_classes=len(train_ds.label_list))

tokenizer = AutoTokenizer.from_pretrained("bert-wwm-chinese")

def convert_example(example, tokenizer):
    encoded_inputs = tokenizer(text=example["text"], max_seq_len=512, pad_to_max_seq_len=True)
    return tuple([np.array(x, dtype="int64") for x in [
            encoded_inputs["input_ids"], encoded_inputs["token_type_ids"], [example["label"]]]])
train_ds = train_ds.map(partial(convert_example, tokenizer=tokenizer))

batch_sampler = paddle.io.BatchSampler(dataset=train_ds, batch_size=8, shuffle=True)
train_data_loader = paddle.io.DataLoader(dataset=train_ds, batch_sampler=batch_sampler, return_list=True)

optimizer = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters())

criterion = paddle.nn.loss.CrossEntropyLoss()

# 使用训练数据进行一次训练，也就是Epoch=1，使用思路和PyTorch类似，但不同。
for input_ids, token_type_ids, labels in train_data_loader():
    logits = model(input_ids, token_type_ids)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.clear_grad()