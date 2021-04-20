from tqdm import tqdm
import json
import pickle
import torch.utils.data
import random

import db
import config
from bean import *


def build_nn_select_dataset():
    size = config.nn_select_dataset_size
    citations = db.random_sample_citations(size, use_tqdm=True)
    dataset = []
    for citation in tqdm(citations):
        doc_id = citation["src"]
        cite_id = citation["dest"]
        invalid_ids = set(db.query_by_id(doc_id).get("citations", []))
        invalid_ids.add(doc_id)
        while True:
            neg_id = db.random_sample(1)[0]["_id"]
            if neg_id not in invalid_ids:
                break
        dataset.append(NNSelectSample(
            doc_id=doc_id,
            pos_id=cite_id,
            neg_id=neg_id,
        ))
    with open(config.nn_select_dataset_path, "wb") as fp:
        pickle.dump(dataset, fp)


def get_nn_select_dataloaders():
    with open(config.nn_select_dataset_path, "rb") as fp:
        dataset_list = pickle.load(fp)
    random.shuffle(dataset_list)
    N = len(dataset_list)
    test_size = int(N * config.test_set_ratio)
    test_set = dataset_list[:test_size]
    train_set = dataset_list[test_size:]
    return torch.utils.data.DataLoader(
        NNSelectDataset(train_set),
        batch_size=config.batch_size,
        shuffle=True,
    ), torch.utils.data.DataLoader(
        NNSelectDataset(test_set),
        batch_size=config.batch_size,
        shuffle=False,
    )


class NNSelectDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_list):
        self.dataset_list = dataset_list

    def __getitem__(self, index: int):
        item = self.dataset_list[index]
        doc_id, pos_id, neg_id = item.doc_id, item.pos_id, item.neg_id
        doc = db.query_by_id(doc_id)
        pos = db.query_by_id(pos_id)
        neg = db.query_by_id(neg_id)
        return NNSelectRichSample(
            doc_title=doc.get("title", ""),
            doc_abstract=doc.get("abstract", ""),
            pos_title=pos.get("title", ""),
            pos_abstract=pos.get("abstract", ""),
            pos_cited_cnt=pos.get("cited_cnt", 0),
            neg_title=neg.get("title", ""),
            neg_abstract=neg.get("abstract", ""),
            neg_cited_cnt=neg.get("cited_cnt", 0),
        )

    def __len__(self):
        return len(self.dataset_list)


def get_triple_batch(batch_size=-1):
    if batch_size == -1:
        batch_size = config.batch_size


if __name__ == '__main__':
    build_nn_select_dataset()
