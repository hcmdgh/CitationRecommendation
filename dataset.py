import config
import db
from tqdm import tqdm
from collections import namedtuple
import torch
import torch.utils.data
import pickle

Sample = namedtuple("Sample", "query_i, other_i, is_pos")
RichSample = namedtuple("RichSample", "query_doc, other_doc, is_pos")
Paper = namedtuple("Paper", "i, title, abstract, venue, keywords, in_cnt, out_cnt")


class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        with open(config.dataset_path, "rb") as fp:
            self.dataset_list = pickle.load(fp)
        with open(config.lookup_table_path, "rb") as fp:
            self.lookup_table = pickle.load(fp)

    def __getitem__(self, index: int):
        sample = self.dataset_list[index]
        return RichSample(
            query_doc=self.lookup_table[sample.query_i],
            other_doc=self.lookup_table[sample.other_i],
            is_pos=sample.is_pos,
        )

    def __len__(self):
        return len(self.dataset_list)


def get_dataloader():
    return torch.utils.data.DataLoader(
        MyDataset(),
        batch_size=config.batch_size,
        shuffle=True,
    )


# def get_dataloader_task1():
#     if not os.path.isfile(config.dataset_pkl_path):
#         dataset = []
#         with open(config.dataset_path, "r", encoding="utf-8") as fp:
#             # MAX_SIZE = 10000
#             for line in tqdm(fp, desc="加载数据集"):
#                 # if MAX_SIZE <= 0: break
#                 # MAX_SIZE -= 1
#                 doc_i, pos_i, neg_i = map(int, line.strip().split())
#                 doc = db.query_by_i(doc_i)
#                 pos = db.query_by_i(pos_i)
#                 neg = db.query_by_i(neg_i)
#                 dataset.append(DataItemTask1(
#                     doc_cite=len(doc["inCitations"]),
#                     pos_cite=len(pos["inCitations"]),
#                     neg_cite=len(neg["inCitations"]),
#                     doc_title=doc["title"],
#                     doc_abstract=doc["paperAbstract"],
#                     pos_title=pos["title"],
#                     pos_abstract=pos["paperAbstract"],
#                     neg_title=neg["title"],
#                     neg_abstract=neg["paperAbstract"],
#                 ))
#         with open(config.dataset_pkl_path, "wb") as fp:
#             pickle.dump(dataset, fp)
#     else:
#         with open(config.dataset_pkl_path, "rb") as fp:
#             dataset = pickle.load(fp)
#     return torch.utils.data.DataLoader(
#         dataset,
#         batch_size=config.batch_size,
#         shuffle=True,
#     )
#
#
# def build_dataset_task1():
#     with open(config.dataset_path, "w", encoding="utf-8") as fp:
#         paper_cnt = config.dataset_paper_cnt
#         neg_cnt = config.negative_samples_per_pair
#         for paper in tqdm(db.random_sample(paper_cnt, use_tqdm=True)):
#             paper_id = paper["id"]
#             paper_i = paper["i"]
#             pos_ids = set(paper["outCitations"])
#             neg_ids = []
#             for i in range(neg_cnt):
#                 while True:
#                     neg_paper = db.random_sample(1)
#                     neg_id = neg_paper[0]["id"]
#                     if neg_id != paper_id and neg_id not in pos_ids:
#                         break
#                 neg_ids.append(neg_id)
#             for pos_id in pos_ids:
#                 try:
#                     pos_i = db.query_by_id(pos_id)["i"]
#                 except FileNotFoundError:
#                     continue
#                 for neg_id in neg_ids:
#                     neg_i = db.query_by_id(neg_id)["i"]
#                     fp.write(f"{paper_i}\t{pos_i}\t{neg_i}\n")


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
