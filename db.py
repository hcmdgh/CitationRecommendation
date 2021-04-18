import pymongo
import config
import random
from tqdm import tqdm

_conn = pymongo.MongoClient()
_paper = _conn.citation_recommendation.paper
_cache_id = dict()
_cache_i = dict()


def query_by_id(id_):
    if id_ not in _cache_id:
        paper = _paper.find_one({ "id": id_ })
        if paper is None:
            raise FileNotFoundError(f"未找到此id：{id_}")
        if config.use_db_cache:
            _cache_id[id_] = paper
        return paper
    return _cache_id[id_]


def query_by_i(i):
    if i not in _cache_i:
        paper = _paper.find_one({ "i": i })
        if paper is None:
            raise FileNotFoundError(f"未找到此i：{i}")
        if config.use_db_cache:
            _cache_i[i] = paper
        return paper
    return _cache_i[i]


def random_sample(cnt, use_tqdm=False):
    papers = []
    visited = set()
    for _ in tqdm(range(cnt), desc="sampling") if use_tqdm else range(cnt):
        while True:
            i = random.randrange(config.corpus_size)
            if i not in visited:
                visited.add(i)
                break
        paper = query_by_i(i)
        papers.append(paper)
    return papers


if __name__ == '__main__':
    total = error = 0
    for paper in _paper.find():
        cite_ids = paper["outCitations"]
        total += len(cite_ids)
        for cite_id in cite_ids:
            try:
                query_by_id(cite_id)
            except FileNotFoundError:
                # print(cite_id)
                error += 1
                if error % 1000 == 0:
                    print(f"total: {total} error: {error} rate: {error / total}")
    print(f"total: {total} error: {error} rate: {error / total}")
