# -*- coding: utf-8 -*-
# weibifan 2022-10-11
# 百度智能云，商业化产品，包括WebAPI和本地SDK，都需要鉴权。
# 百度账号在百度智能云里需要再次认证。

'''
任务：语音识别，OCR，人脸识别，人体识别，图像识别，图像检索，内容审核
自然语言处理，机器翻译

每种任务都有相关的API Key和 Secret Key

1）先创建应用，然后就获取了AppID和两个key
2）依据两个Key，获取access_token，有效期1个月
3）依据access_token，调用接口。


https://ai.baidu.com/ai-doc/FACE/Lk37c1tpf
这个文档的第2个例子是错的。

'''

import requests



# 第2步：
# client_id 为官网获取的AK， client_secret 为官网获取的SK
# weibifan 下面字符串不能有空格！！！！！！！！！！！！！！！！！
host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=l4p49REHiMaQ7MH668g4qjGC&client_secret=v6017lqjpgE4SYnZun0hGIu6AVLhay1O'
response = requests.get(host)
if response:
    print(response.json())
else:
    print('unknown error')

'''
{'refresh_token': '25.cb9ffb868c94c243d550e27a6c71c93b.315360000.1980857093.282335-27860817', 'expires_in': 2592000, 'session_key': '9mzdDtMJeqjLwhjApdwtoyqaJLYdMlitNobe0C7IhNdldJETFY/jSdI/el6zKE3XZTWUA/6UNs25+4FyfmMQyes4e2CXQw==', 'access_token': '24.5a5bd852da5bf181b20e1f179f0c09dd.2592000.1668089093.282335-27860817', 'scope': 'public brain_all_scope vis-faceverify_faceverify_h5-face-liveness vis-faceverify_FACE_V3 vis-faceverify_idl_face_merge vis-faceverify_FACE_EFFECT vis-faceverify_face_beauty vis-faceverify_face_feature_sdk brain_face_scene_scope wise_adapt lebo_resource_base lightservice_public hetu_basic lightcms_map_poi kaidian_kaidian ApsMisTest_Test权限 vis-classify_flower lpq_开放 cop_helloScope ApsMis_fangdi_permission smartapp_snsapi_base smartapp_mapp_dev_manage iop_autocar oauth_tp_app smartapp_smart_game_openapi oauth_sessionkey smartapp_swanid_verify smartapp_opensource_openapi smartapp_opensource_recapi fake_face_detect_开放Scope vis-ocr_虚拟人物助理 idl-video_虚拟人物助理 smartapp_component smartapp_search_plugin avatar_video_test b2b_tp_openapi b2b_tp_openapi_online smartapp_gov_aladin_to_xcx', 'session_secret': 'e87781942bbaa364ca13574ff784a239'}
'''

# 第3步：


