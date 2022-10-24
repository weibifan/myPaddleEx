# -*- coding: utf-8 -*-
# weibifan 2022-10-4
# 中文语法纠错
# 英文语法纠错：Huggingface  tuner007/pegasus_paraphrase
r"""

https://bj.bcebos.com/paddlenlp/taskflow/text_correction/ernie-csc/model_state.pdparams

[2022-10-24 23:31:53,310] [    INFO] - Already cached C:\Users\Wei\.paddlenlp\models\ernie-1.0\vocab.txt
[2022-10-24 23:31:53,318] [    INFO] - tokenizer config file saved in C:\Users\Wei\.paddlenlp\models\ernie-1.0\tokenizer_config.json
[2022-10-24 23:31:53,318] [    INFO] - Special tokens file saved in C:\Users\Wei\.paddlenlp\models\ernie-1.0\special_tokens_map.json

"""

import paddlehub as hub

# Load ernie-csc
module = hub.Module(name="ernie-csc")

# String input
results = module.predict("遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。")
print(results)
# [{'source': '遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。', 'target': '遇到逆境时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。', 'errors': [{'position': 3, 'correction': {'竟': '境'}}]}]

results = module.predict("我们才能朝著成功之路前进。")   #朝着  没有识别处理
print(results)

# List input
results = module.predict(['遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。', '人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。'])
print(results)
# [{'source': '遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。', 'target': '遇到逆境时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。', 'errors': [{'position': 3, 'correction': {'竟': '境'}}]}, {'source': '人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。', 'target': '人生就是如此，经过磨练才能让自己更加茁壮，才能使自己更加乐观。', 'errors': [{'position': 18, 'correction': {'拙': '茁'}}]}]