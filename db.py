import pymongo
import config
import random
from tqdm import tqdm

_conn = pymongo.MongoClient()
_paper = _conn.citation_recommendation.sheared_paper
_citation = _conn.citation_recommendation.citation
_cache_id = dict()
_cache_i = dict()
_paper_cnt = _paper.count_documents({})
_citation_cnt = _citation.count_documents({})


def query_by_id(id_):
    if id_ not in _cache_id:
        paper = _paper.find_one({ "_id": id_ })
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
            i = random.randrange(_paper_cnt)
            if i not in visited:
                visited.add(i)
                break
        paper = query_by_i(i)
        papers.append(paper)
    return papers


def random_sample_citations(cnt, use_tqdm=False):
    citations = []
    visited = set()
    for _ in tqdm(range(cnt), desc="sampling") if use_tqdm else range(cnt):
        while True:
            i = random.randrange(_citation_cnt)
            if i not in visited:
                visited.add(i)
                break
        citation = _citation.find_one({ "_id": i })
        citations.append(citation)
    return citations


def map_papers(func):
    for paper in tqdm(_paper.find()):
        new_paper = func(paper)
        if new_paper:
            _paper.save(new_paper)


if __name__ == '__main__':
    res = random_sample(1000)
    print(res[:3])
    res = random_sample_citations(1000)
    print(res[:3])
