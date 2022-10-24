# -*- coding: utf-8 -*-
# weibifan 2022-10-5
# 中文句法解析  废弃

'''

ModuleNotFoundError: No module named 'ddparser'


ddparser
LAC

'''
import cv2
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