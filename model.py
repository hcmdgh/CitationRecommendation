import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertConfig
from util import *


class BertEmbedding(nn.Module):
    def __init__(self):
        super(BertEmbedding, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel(BertConfig(hidden_size=config.bert_hidden_size))

    def forward(self, text_batch):
        tokens = self.tokenizer(text_batch, return_tensors="pt", padding=True)
        tokens = clip_tokens(tokens)
        output = self.model(**tokens).last_hidden_state
        output = torch.mean(output, dim=1)
        return output


class NNSelect(nn.Module):
    def __init__(self):
        super(NNSelect, self).__init__()
        self.bert_model = BertEmbedding()
        self.lambda_ = nn.Parameter(torch.randn(2))

    def forward(self, title_batch, abstract_batch):
        title_embed = self.bert_model(title_batch)
        abstract_embed = self.bert_model(abstract_batch)

        softmax_lambda = F.softmax(self.lambda_, dim=0)
        lambda_title, lambda_abstract = softmax_lambda

        output = lambda_title * title_embed + lambda_abstract * abstract_embed
        return output
