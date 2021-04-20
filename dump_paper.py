import pymongo
from tqdm import tqdm

conn = pymongo.MongoClient()
collection = conn.citation_recommendation.paper
collection.drop()

with open(r"D:\Desktop\毕业设计\数据集\citation-network1\outputacm.txt", "r", encoding="utf-8") as fp:
    paper = dict()
    total_cnt = 0
    actual_cnt = 0
    for line in tqdm(fp):
        line = line.strip()
        if line.startswith("#*"):
            title = line[2:]
            paper["title"] = title
        elif line.startswith("#@"):
            authors = line[2:]
            paper["authors"] = authors
        elif line.startswith("#t"):
            year = int(line[2:])
            paper["year"] = year
        elif line.startswith("#c"):
            venue = line[2:]
            paper["venue"] = venue
        elif line.startswith("#index"):
            id_ = int(line[6:])
            paper["_id"] = id_
        elif line.startswith("#%"):
            cite_id = int(line[2:])
            if "citations" not in paper:
                paper["citations"] = []
            paper["citations"].append(cite_id)
        elif line.startswith("#!"):
            abstract = line[2:]
            paper["abstract"] = abstract
        else:
            if paper:
                total_cnt += 1
                collection.insert_one(paper)
                actual_cnt += 1
            paper.clear()
    print(actual_cnt)
    print(total_cnt)
