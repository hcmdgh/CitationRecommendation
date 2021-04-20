import config
import db
from tqdm import tqdm
from collections import namedtuple
import torch
import torch.utils.data
import pickle
import random
from bean import *


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_list, lookup_table):
        self.dataset_list = dataset_list
        self.lookup_table = lookup_table

    def __getitem__(self, index: int):
        sample = self.dataset_list[index]
        query_doc = self.lookup_table[sample.query_i]
        other_doc = self.lookup_table[sample.other_i]
        return RichSample(
            query_title=query_doc.title,
            query_abstract=query_doc.abstract,
            query_venue=query_doc.venue,
            query_keywords=' '.join(query_doc.keywords),
            query_in_cnt=float(query_doc.in_cnt),
            other_title=other_doc.title,
            other_abstract=other_doc.abstract,
            other_venue=other_doc.venue,
            other_keywords=' '.join(other_doc.keywords),
            other_in_cnt=float(other_doc.in_cnt),
            target=float(sample.is_pos),
        )

    def __len__(self):
        return len(self.dataset_list)


def get_dataloaders():
    with open(config.dataset_path, "rb") as fp:
        dataset_list = pickle.load(fp)
    with open(config.lookup_table_path, "rb") as fp:
        lookup_table = pickle.load(fp)
    random.shuffle(dataset_list)
    N = len(dataset_list)
    test_size = int(N * config.test_set_ratio)
    test_set = dataset_list[:test_size]
    train_set = dataset_list[test_size:]
    return torch.utils.data.DataLoader(
        MyDataset(dataset_list=train_set, lookup_table=lookup_table),
        batch_size=config.batch_size,
        shuffle=True,
    ), torch.utils.data.DataLoader(
        MyDataset(dataset_list=test_set, lookup_table=lookup_table),
        batch_size=config.batch_size,
        shuffle=False,
    )


def build_dataset(doc_cnt):
    dataset = []
    for doc in tqdm(db.random_sample(doc_cnt, use_tqdm=True)):
        doc_i = doc["i"]
        invalid_ids = set(doc["inCitations"] + doc["outCitations"])
        invalid_ids.add(doc["id"])
        pos_cnt = 0

        # build pos set
        for pos_id in doc["outCitations"]:
            try:
                pos_i = db.query_by_id(pos_id)["i"]
                pos_cnt += 1
                dataset.append(Sample(query_i=doc_i, other_i=pos_i, is_pos=True))
            except FileNotFoundError:
                pass

        # build neg set
        for i in range(pos_cnt):
            while True:
                neg_doc = db.random_sample(1)[0]
                neg_id = neg_doc["id"]
                if neg_id not in invalid_ids:
                    break
            invalid_ids.add(neg_id)
            dataset.append(Sample(query_i=doc_i, other_i=neg_doc["i"], is_pos=False))
    with open(config.dataset_path, "wb") as fp:
        pickle.dump(dataset, fp)

    lookup_table = dict()
    for sample in tqdm(dataset):
        for doc_i in [sample.query_i, sample.other_i]:
            if doc_i not in lookup_table:
                paper = db.query_by_i(doc_i)
                lookup_table[doc_i] = Paper(
                    i=paper["i"],
                    title=paper["title"],
                    abstract=paper["paperAbstract"],
                    venue=paper["venue"],
                    keywords=paper["keyPhrases"],
                    in_cnt=len(paper["inCitations"]),
                    out_cnt=len(paper["outCitations"]),
                )
    with open(config.lookup_table_path, "wb") as fp:
        pickle.dump(lookup_table, fp)

    print(f"构建完毕！数据集大小：{len(dataset)}")
    print(f"查询字典大小：{len(lookup_table)}")


if __name__ == '__main__':
    build_dataset(doc_cnt=3000)
    # x, y = get_dataloaders()
    # print(x, y)
