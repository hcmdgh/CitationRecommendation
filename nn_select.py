import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os

import dataset
from model import *


def loss_fn(doc_embed, pos_embed, neg_embed, pos_cited_cnt, neg_cited_cnt):
    def B(cite):
        return to_gpu(torch.sigmoid(cite / 100.) / 50.)

    return torch.mean(F.relu(to_gpu(config.alpha)
                      + torch.cosine_similarity(doc_embed, neg_embed, dim=1) + B(neg_cited_cnt)
                      - torch.cosine_similarity(doc_embed, pos_embed, dim=1) - B(pos_cited_cnt)))


def test(model, test_loader):
    model.eval()
    total_loss = 0.
    total_cnt = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="testing"):
            doc_embed = model(title_batch=batch.doc_title, abstract_batch=batch.doc_abstract)
            pos_embed = model(title_batch=batch.pos_title, abstract_batch=batch.pos_abstract)
            neg_embed = model(title_batch=batch.neg_title, abstract_batch=batch.neg_abstract)
            loss = loss_fn(
                doc_embed=doc_embed,
                pos_embed=pos_embed,
                neg_embed=neg_embed,
                pos_cited_cnt=batch.pos_cited_cnt,
                neg_cited_cnt=batch.neg_cited_cnt,
            )
            total_loss += float(loss)
            total_cnt += 1
    return total_loss / total_cnt


def main():
    train_loader, test_loader = dataset.get_nn_select_dataloaders()
    print_plus("len train_loader:", len(train_loader))
    print_plus("len test_loader:", len(test_loader))
    model = to_gpu(NNSelect())
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
            doc_embed = model(title_batch=batch.doc_title, abstract_batch=batch.doc_abstract)
            pos_embed = model(title_batch=batch.pos_title, abstract_batch=batch.pos_abstract)
            neg_embed = model(title_batch=batch.neg_title, abstract_batch=batch.neg_abstract)
            loss = loss_fn(
                doc_embed=doc_embed,
                pos_embed=pos_embed,
                neg_embed=neg_embed,
                pos_cited_cnt=batch.pos_cited_cnt,
                neg_cited_cnt=batch.neg_cited_cnt,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss)
            total_cnt += 1
            if step % 100 == 0:
                print_plus(f"epoch: {epoch} step: {step} avg_loss: {total_loss / total_cnt}")
            if step % 1000 == 0:
                test_avg_loss = test(model, test_loader)
                print_plus(f"test acc: {test_avg_loss}")

        print_plus(f"epoch: {epoch} avg_loss: {total_loss / total_cnt}")

        # 保存模型参数
        torch.save(model.state_dict(), os.path.join(config.model_state_path, f"model_epoch_{epoch}.pt"))


if __name__ == '__main__':
    main()
