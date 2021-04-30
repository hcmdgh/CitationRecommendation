import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertConfig
from tqdm import tqdm

import dataset_old
from util import *
from model import *
import config


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
    train_loader, test_loader = dataset_old.get_dataloaders()
    print_plus("len train_loader:", len(train_loader))
    print_plus("len test_loader:", len(test_loader))
    model = to_gpu(NNRank())
    print_plus("total params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.L2_penalty,
    )

    model.train()
    for epoch in range(config.epoch):
        total_loss = 0.
        total_cnt = 0
        for step, batch in enumerate(train_loader):
            y_pred = model(batch)
            y_true = to_gpu(batch.target)
            loss = loss_fn(y_true=y_true, y_pred=y_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss)
            total_cnt += 1
            if step % 100 == 0:
                print_plus(f"epoch: {epoch} step: {step} avg_loss: {total_loss / total_cnt}")
            if step % 1000 == 0:
                correct, total = test(model, test_loader)
                print_plus(f"test acc: {correct / total}")

        print_plus(f"epoch: {epoch} avg_loss: {total_loss / total_cnt}")


if __name__ == '__main__':
    main()
