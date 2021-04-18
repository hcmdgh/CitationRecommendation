import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertConfig
import config
import db
import dataset
from util import *


# [input]
#   title_batch: [batch_size, sentence_len]
#   abstract_batch: [batch_size, sentence_len]
# [output]
#   output: [batch_size, embed_size]
class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_config = BertConfig(hidden_size=84)
        # self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel(bert_config)
        self.lambda_ = nn.Parameter(torch.randn(2))

    def forward(self, title_batch, abstract_batch):
        # [batch_size, sentence_len] -> [batch_size, sentence_len, embed_size]
        title_inputs = self.bert_tokenizer(title_batch, return_tensors="pt", padding=True)
        title_inputs = clip_tokens(title_inputs)
        title_outputs = self.bert_model(**title_inputs).last_hidden_state

        # [batch_size, sentence_len] -> [batch_size, sentence_len, embed_size]
        abstract_inputs = self.bert_tokenizer(abstract_batch, return_tensors="pt", padding=True)
        abstract_inputs = clip_tokens(abstract_inputs)
        abstract_outputs = self.bert_model(**abstract_inputs).last_hidden_state

        # [batch_size, sentence_len, embed_size] -> [batch_size, embed_size]
        title_embed = title_outputs.mean(1)
        abstract_embed = abstract_outputs.mean(1)

        softmax_lambda = F.softmax(self.lambda_, dim=0)
        lambda_title, lambda_abstract = softmax_lambda

        output = lambda_title * title_embed + lambda_abstract * abstract_embed
        return output


def loss_fn(doc_embed, pos_embed, neg_embed, pos_cite, neg_cite):
    def B(cite):
        return to_gpu(torch.sigmoid(cite / 100.) / 50.)

    return torch.mean(F.relu(config.alpha
                      + torch.cosine_similarity(doc_embed, neg_embed, dim=1) + B(neg_cite)
                      - torch.cosine_similarity(doc_embed, pos_embed, dim=1) - B(pos_cite)))


def main():
    train_loader = dataset.get_dataloader_task1()
    print("batch per epoch:", len(train_loader))
    model = to_gpu(Embedding())
    print("total params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.L2_penalty,
    )

    model.train()
    for epoch in range(config.epoch):
        total_loss = 0.
        for step, batch in enumerate(train_loader):
            doc_embed = model(batch.doc_title, batch.doc_abstract)
            pos_embed = model(batch.pos_title, batch.pos_abstract)
            neg_embed = model(batch.neg_title, batch.neg_abstract)
            loss = loss_fn(
                doc_embed=doc_embed,
                pos_embed=pos_embed,
                neg_embed=neg_embed,
                pos_cite=batch.pos_cite,
                neg_cite=batch.neg_cite,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 20 == 0:
                print(f"epoch: {epoch} step: {step} loss: {float(loss)}")
            total_loss += float(loss)

        avg_loss = total_loss / len(train_loader)
        print(f"epoch: {epoch} avg_loss: {avg_loss}")


if __name__ == '__main__':
    main()
    # model = Embedding()
    # out = model(
    #     title_batch=[
    #         "Independence of Containing Patterns Property and Its Application in Tree Pattern Query Rewriting Using Views",
    #         "Caregiver status affects medication adherence among older home care clients with heart failure.",
    #         "An Automated Negotiation Engine for Consistency of Access Policies in Grid",
    #     ],
    #     abstract_batch=[
    #         "Optical Three-axis Tactile Sensor for Robotic Fingers",
    #         "A golden age for malaria research and innovation",
    #         "A natural fast-cleaving branching ribozyme from the amoeboflagellate Naegleria pringsheimi.",
    #     ],
    # )
    # print(out.shape)
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
