# coding: utf-8
# author: JayChan
import torch
import torch.nn as nn
from transformers import AutoTokenizer,AutoModel,AutoConfig
class Model(nn.Module):


    def __init__(self, args):
        super(Model, self).__init__()
        # 注意使用simCSE时，要手动调整配置文件中的dropout_prob，或者下面这条语句增加dropout_prob属性。
        self.config = AutoConfig.from_pretrained(args.ptm_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.ptm_path)
        # self.bert = AutoModel.from_pretrained(args.ptm_path,config=self.config) # 加载权重，并在此基础之上train
        self.bert = AutoModel.from_config(config=self.config) # 根据自定义的config从头开始train一个模型
        for param in self.bert.base_model.parameters():
            param.requires_grad = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.max_len = args.max_length


    def forward(self, x):
        text = x
        pt_batch =self.tokenizer(list(text),
                                 padding = True,
                                 truncation = True,
                                 max_length = self.max_len,
                                 return_tensors = "pt").to(self.device)

        outputs = self.bert(**pt_batch,output_hidden_states=False, output_attentions=False)
        return outputs['pooler_output']