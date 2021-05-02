import pymongo
from tqdm import tqdm

conn = pymongo.MongoClient()
paper_collection = conn.citation_recommendation.sheared_paper
citation_collection = conn.citation_recommendation.citation

idx = 0
citation_collection.drop()
for paper in tqdm(paper_collection.find()):
    citations = paper.get("citations", [])
    paper_id = paper["_id"]
    for cite_id in citations:
        citation_collection.insert_one({
            "_id": idx,
            "src": paper_id,
            "dest": cite_id,
        })
        idx += 1
