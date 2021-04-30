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


class NNRank(nn.Module):
    def __init__(self):
        super(NNRank, self).__init__()
        self.bert_embedding = BertEmbedding()
        self.fc1 = nn.Linear(5, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 1)

    def forward(self, batch):
        query_title_embed = self.bert_embedding(batch.query_title)
        other_title_embed = self.bert_embedding(batch.other_title)
        title_similarity = torch.cosine_similarity(query_title_embed, other_title_embed, dim=1)

        query_abstract_embed = self.bert_embedding(batch.query_abstract)
        other_abstract_embed = self.bert_embedding(batch.other_abstract)
        abstract_similarity = torch.cosine_similarity(query_abstract_embed, other_abstract_embed, dim=1)

        query_venue_embed = self.bert_embedding(batch.query_venue)
        other_venue_embed = self.bert_embedding(batch.other_venue)
        venue_similarity = torch.cosine_similarity(query_venue_embed, other_venue_embed, dim=1)

        query_keywords_embed = self.bert_embedding(batch.query_keywords)
        other_keywords_embed = self.bert_embedding(batch.other_keywords)
        keywords_similarity = torch.cosine_similarity(query_keywords_embed, other_keywords_embed, dim=1)

        log_in_cnt = torch.max(torch.log(to_gpu(batch.other_in_cnt)), to_gpu(-1))

        concat = torch.stack(
            [title_similarity, abstract_similarity, venue_similarity, keywords_similarity, log_in_cnt],
            dim=1,
        )

        output = F.elu(self.fc1(concat))
        output = F.elu(self.fc2(output))
        output = torch.sigmoid(self.fc3(output))

        return output
