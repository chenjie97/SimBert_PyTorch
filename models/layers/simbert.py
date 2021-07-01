# coding: utf-8
# author: JayChan
import torch
import torch.nn as nn
from transformers import AutoTokenizer,AutoModel,AutoConfig
import numpy as np
class Model(nn.Module):


    def __init__(self, args):
        super(Model, self).__init__()
        self.config = AutoConfig.from_pretrained(args.ptm_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.ptm_path)
        # self.bert = AutoModel.from_pretrained(args.ptm_path,config=self.config) # 加载权重，并在此基础之上train
        self.bert = AutoModel.from_config(config=self.config) # 根据自定义的config从头开始train一个模型
        for param in self.bert.base_model.parameters():
            param.requires_grad = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.max_len = args.max_length
        self.linear = nn.Linear(self.config.hidden_size,self.config.vocab_size)


    def forward(self, x):
        text,synonym = x
        pt_batch =self.tokenizer(list(text),list(synonym),
                                 padding = True,
                                 truncation = True,
                                 max_length = self.max_len,
                                 return_tensors = "pt").to(self.device)
        # context = pt_batch['input_ids']  # 输入的句子
        # mask = pt_batch['attention_mask']  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        # print(self.bert)
        # outputs = self.bert(context, attention_mask=mask)
        # torch.set_printoptions(threshold=np.inf)
        # print(pt_batch)
        outputs = self.bert(**pt_batch,output_hidden_states=False, output_attentions=False)
        out_cls = outputs['pooler_output']
        out_seq = self.linear(outputs['last_hidden_state'])


        return out_cls,out_seq,pt_batch