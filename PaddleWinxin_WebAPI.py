# -*- coding: utf-8 -*-
# weibifan 2022-10-6
# 文心大模型，百度官方的PLMs，与PaddleHub形成竞争关系
# 基于ERNIE3，对中文支持好一些。https://wenxin.baidu.com/

'''
基于ERNIE3构建的各类task：包括NLP，CV，跨模态等领域，医疗、金融等领域
相关论文：https://github.com/PaddlePaddle/ERNIE

第1种方法：在线使用公共API，只有NLP（ERNIE 3.0 文本理解与创作），多模态（ERNIE-ViLG 文生图）及PLATO。

第2种方法：使用开发套件进行开发，分为3类。
1）大模型套件ERNIEKit，开源套件，基于EINIE3大模型的微调接口
①开源版。这个很难使用。将各种PLM下载到本地来使用。涵盖了NLP大模型和跨模态大模型。
②旗舰版，需要购买在线服务器。

2）专业版套件（EasyDL）：微调PLMs，本地或云端微调，可进行本地或云部署。收费套件。
3）企业版套件（BML）：


'''
# 第1种方法：调用公开API
import wenxin_api # 可以通过"pip install wenxin-api"命令安装
from wenxin_api.tasks.summarization import Summarization
wenxin_api.ak = "vI7szfcGlsPoc4kPbzB4Gj7ZvYUDMlSc"
wenxin_api.sk = "7Nk22OqxvGmmfmz47GhpP40sSHmMejYH"
input_dict = {
    "text": "文章：外媒7月18日报道，阿联酋政府当日证实该国将建设首个核电站，以应对不断上涨的用电需求。分析称阿联酋作为世界第三大石油出口国，更愿意将该能源用于出口，而非发电。首座核反应堆预计在2017年运行。cntv李婉然编译报道\n摘要：",
    "seq_len": 512,
    "topp": 0.3,
    "penalty_score": 1.0,
    "min_dec_len": 4,
    "is_unidirectional": 0,
    "task_prompt": "Summarization"
}
rst = Summarization.create(**input_dict)
print(rst)