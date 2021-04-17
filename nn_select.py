import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertConfig


# [input]
#   title_batch: [batch_size, sentence_len]
#   abstract_batch: [batch_size, sentence_len]
# [output]
#   output: [batch_size, embed_size]
class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.lambda_title = nn.Parameter(torch.randn(()))
        self.lambda_abstract = nn.Parameter(torch.randn(()))

    def forward(self, title_batch, abstract_batch):
        # [batch_size, sentence_len] -> [batch_size, sentence_len, embed_size]
        title_inputs = self.bert_tokenizer(title_batch, return_tensors="pt", padding=True)
        title_outputs = self.bert_model(**title_inputs).last_hidden_state

        # [batch_size, sentence_len] -> [batch_size, sentence_len, embed_size]
        abstract_inputs = self.bert_tokenizer(abstract_batch, return_tensors="pt", padding=True)
        abstract_outputs = self.bert_model(**abstract_inputs).last_hidden_state

        # [batch_size, sentence_len, embed_size] -> [batch_size, embed_size]
        title_embed = title_outputs.mean(1)
        abstract_embed = abstract_outputs.mean(1)

        output = self.lambda_title * title_embed + self.lambda_abstract * abstract_embed
        return output


if __name__ == '__main__':
    model = Embedding()
    out = model(
        title_batch=[
            "Independence of Containing Patterns Property and Its Application in Tree Pattern Query Rewriting Using Views",
            "Caregiver status affects medication adherence among older home care clients with heart failure.",
            "An Automated Negotiation Engine for Consistency of Access Policies in Grid",
        ],
        abstract_batch=[
            "Optical Three-axis Tactile Sensor for Robotic Fingers",
            "A golden age for malaria research and innovation",
            "A natural fast-cleaving branching ribozyme from the amoeboflagellate Naegleria pringsheimi.",
        ],
    )
    print(out.shape)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
