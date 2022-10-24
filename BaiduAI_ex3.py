# -*- coding: utf-8 -*-
# weibifan 2022-10-11
# 百度智能云，商业化产品，包括WebAPI和本地SDK，都需要鉴权。
# 百度账号在百度智能云里需要再次认证。

'''
任务：语音识别，OCR，人脸识别，人体识别，图像识别，图像检索，内容审核
自然语言处理，机器翻译

每种任务都有相关的API Key和 Secret Key

1）先创建应用，然后就获取了AppID和两个key
2）依据两个Key，获取access_token
3）依据access_token，调用接口。


https://ai.baidu.com/ai-doc/FACE/Lk37c1tpf
这个文档的第2个例子是错的。



'''


# encoding:utf-8

import requests


'''
人脸对比
'''

request_url = "https://aip.baidubce.com/rest/2.0/face/v3/match"

params = "[{\"image\": \"sfasq35sadvsvqwr5q\", \"image_type\": \"BASE64\", \"face_type\": \"LIVE\", \"quality_control\": \"LOW\"},{\"image\": \"sfasq35sadvsvqwr5q\", \"image_type\": \"BASE64\", \"face_type\": \"LIVE\", \"quality_control\": \"LOW\"}]"

params = '[{\"image\":\"https:\/\/baidu-ai.bj.bcebos.com\/face\/faces.jpg\",\"image_type\":\"URL\"},\n{\"image\":\"https:\/\/baidu-ai.bj.bcebos.com\/face\/faces.jpg\",\"image_type\":\"URL\"}]'

access_token = '24.5a5bd852da5bf181b20e1f179f0c09dd.2592000.1668089093.282335-27860817'
request_url = request_url + "?access_token=" + access_token
headers = {'Content-Type': 'application/json;charset=UTF-8'}
response = requests.post(request_url, json=params, headers=headers)
if response:
    print (response.json())
'''  并不是json格式错误，原因未知。
{'error_code': 222200, 'error_msg': 'request body should be json format', 'log_id': 1936209620, 'timestamp': 1665502336, 'cached': 0, 'result': None}
'''



