# -*- coding: utf-8 -*-
# weibifan 2022-10-5
# 中文句法解析  无法升级到高版本

'''
https://www.paddlepaddle.org.cn/hubdetail?name=ddparser&en_category=SyntacticAnalysis

python:>=3.6.0,<3.8.0.
paddlepaddle:>=1.8.2,<2.0
LAC:>=0.1.4

ModuleNotFoundError: No module named 'LAC'
pip install LAC

ModuleNotFoundError: No module named 'ddparser'
pip install ddparser

Exception: Feed list must be given under static mode.

'''
# import cv2
import paddlehub as hub

module = hub.Module(name="ddparser")



test_text = ["百度是一家高科技公司"]
results = module.parse(texts=test_text)  #代码出错
print(results)
'''
test_tokens = [['百度', '是', '一家', '高科技', '公司']]
results = module.parse(texts=test_text, return_visual = True)
print(results)

result = results[0]
data = module.visualize(result['word'],result['head'],result['deprel'])
# or data = result['visual']
cv2.imwrite('test.jpg',data)

'''