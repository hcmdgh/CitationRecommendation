import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertConfig
from tqdm import tqdm

import dataset
from util import *
from bean import *
import config


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


def loss_fn(y_true, y_pred):
    batch_loss = y_true * -torch.log(y_pred) + (to_gpu(1) - y_true) * -torch.log(to_gpu(1) - y_pred)
    return torch.mean(batch_loss)


def test(model, test_loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="testing"):
            y_pred = model(batch)
            y_true = to_gpu(batch.target)
            for i1, i2 in zip(y_true, y_pred):
                if i1 > 0.5 and i2 > 0.5:
                    correct += 1
                elif i1 < 0.5 and i2 < 0.5:
                    correct += 1
                total += 1
    model.train()
    return correct, total


def main():
    train_loader, test_loader = dataset.get_dataloaders()
    print("len train_loader:", len(train_loader))
    print("len test_loader:", len(test_loader))
    model = to_gpu(NNRank())
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
            y_pred = model(batch)
            y_true = to_gpu(batch.target)
            loss = loss_fn(y_true=y_true, y_pred=y_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"epoch: {epoch} step: {step} loss: {float(loss)}")
            if step % 1000 == 0:
                correct, total = test(model, test_loader)
                print(f"test acc: {correct / total}")
            total_loss += float(loss)

        avg_loss = total_loss / len(train_loader)
        print(f"epoch: {epoch} avg_loss: {avg_loss}")


if __name__ == '__main__':
    main()
