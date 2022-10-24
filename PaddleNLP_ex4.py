# -*- coding: utf-8 -*-
# weibifan 2022-10-8
# PaddleNLP，中文自然语言处理的工具，可以完成PLMs的下载，微调，及使用
# https://www.paddlepaddle.org.cn/paddle/paddlenlp
'''
https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_list.html

使用PaddleNLP语义预训练模型ERNIE优化情感分析
https://aistudio.baidu.com/aistudio/projectdetail/1294333
'''
import paddlenlp as ppnlp
from paddlenlp.datasets import load_dataset

train_ds, dev_ds, test_ds = load_dataset("chnsenticorp", splits=["train", "dev", "test"])

print(train_ds.label_list)

for data in train_ds.data[:5]:
    print(data)

'''

'''
MODEL_NAME = "ernie-3.0-medium-zh"

# 实际上是加载了和这个模型一起的另外一个vocab.txt字典文件
# 对于分类任务来说，可以用字建立特征。ernie无法用词
tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained(MODEL_NAME)

ernie_model = ppnlp.transformers.ErnieModel.from_pretrained(MODEL_NAME)


model = ppnlp.transformers.ErnieForSequenceClassification.from_pretrained( \
    MODEL_NAME, num_classes=len(train_ds.label_list))


from functools import partial
from paddlenlp.data import Stack, Tuple, Pad
from utils import  convert_example, create_dataloader

# 模型运行批处理大小
batch_size = 32
max_seq_length = 128

trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length)
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
    Stack(dtype="int64")  # label
): [data for data in fn(samples)]
train_data_loader = create_dataloader(
    train_ds,
    mode='train',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
dev_data_loader = create_dataloader(
    dev_ds,
    mode='dev',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)

from paddlenlp.transformers import LinearDecayWithWarmup

# 训练过程中的最大学习率
learning_rate = 5e-5
# 训练轮次
epochs = 1 #3
# 学习率预热比例
warmup_proportion = 0.1
# 权重衰减系数，类似模型正则项策略，避免模型过拟合
weight_decay = 0.01

num_training_steps = len(train_data_loader) * epochs
lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps, warmup_proportion)
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=weight_decay,
    apply_decay_param_fun=lambda x: x in [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ])

criterion = paddle.nn.loss.CrossEntropyLoss()
metric = paddle.metric.Accuracy()

import paddle.nn.functional as F
from utils import evaluate

global_step = 0
for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):
        input_ids, segment_ids, labels = batch
        logits = model(input_ids, segment_ids)
        loss = criterion(logits, labels)
        probs = F.softmax(logits, axis=1)
        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()

        global_step += 1
        if global_step % 10 == 0 :
            print("global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f" % (global_step, epoch, step, loss, acc))
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()
    evaluate(model, criterion, metric, dev_data_loader)

#model.save_pretrained('checkpoint')
#tokenizer.save_pretrained('checkpoint')

from utils import predict

data = [
    {"text":'这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般'},
    {"text":'怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片'},
    {"text":'作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。'},
]
label_map = {0: 'negative', 1: 'positive'}

results = predict(
    model, data, tokenizer, label_map, batch_size=batch_size)
for idx, text in enumerate(data):
    print('Data: {} \t Lable: {}'.format(text, results[idx]))