import config
import db
from tqdm import tqdm
from collections import namedtuple
import torch
import torch.utils.data
import os
import pickle

DataItem = namedtuple("DataItem", "doc_i pos_i neg_i")
DataItemTask1 = namedtuple("DataItemTask1", "doc_cite doc_title doc_abstract pos_cite pos_title pos_abstract neg_cite neg_title neg_abstract")
DataItemTask2 = namedtuple("DataItemTask2", "")


def get_dataloader_task1():
    if not os.path.isfile(config.dataset_pkl_path):
        dataset = []
        with open(config.dataset_path, "r", encoding="utf-8") as fp:
            # MAX_SIZE = 10000
            for line in tqdm(fp, desc="加载数据集"):
                # if MAX_SIZE <= 0: break
                # MAX_SIZE -= 1
                doc_i, pos_i, neg_i = map(int, line.strip().split())
                doc = db.query_by_i(doc_i)
                pos = db.query_by_i(pos_i)
                neg = db.query_by_i(neg_i)
                dataset.append(DataItemTask1(
                    doc_cite=len(doc["inCitations"]),
                    pos_cite=len(pos["inCitations"]),
                    neg_cite=len(neg["inCitations"]),
                    doc_title=doc["title"],
                    doc_abstract=doc["paperAbstract"],
                    pos_title=pos["title"],
                    pos_abstract=pos["paperAbstract"],
                    neg_title=neg["title"],
                    neg_abstract=neg["paperAbstract"],
                ))
        with open(config.dataset_pkl_path, "wb") as fp:
            pickle.dump(dataset, fp)
    else:
        with open(config.dataset_pkl_path, "rb") as fp:
            dataset = pickle.load(fp)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )


def build_dataset():
    with open(config.dataset_path, "w", encoding="utf-8") as fp:
        paper_cnt = config.dataset_paper_cnt
        neg_cnt = config.negative_samples_per_pair
        for paper in tqdm(db.random_sample(paper_cnt, use_tqdm=True)):
            paper_id = paper["id"]
            paper_i = paper["i"]
            pos_ids = set(paper["outCitations"])
            neg_ids = []
            for i in range(neg_cnt):
                while True:
                    neg_paper = db.random_sample(1)
                    neg_id = neg_paper[0]["id"]
                    if neg_id != paper_id and neg_id not in pos_ids:
                        break
                neg_ids.append(neg_id)
            for pos_id in pos_ids:
                try:
                    pos_i = db.query_by_id(pos_id)["i"]
                except FileNotFoundError:
                    continue
                for neg_id in neg_ids:
                    neg_i = db.query_by_id(neg_id)["i"]
                    fp.write(f"{paper_i}\t{pos_i}\t{neg_i}\n")


if __name__ == '__main__':
    build_dataset()
