import pymongo
from tqdm import tqdm

conn_buaa = pymongo.MongoClient()
paper_buaa = conn_buaa.citation_recommendation.paper
citation_buaa = conn_buaa.citation_recommendation.citation

conn_ali = pymongo.MongoClient(host="39.102.32.190")
paper_ali = conn_ali.citation_recommendation.paper
citation_ali = conn_ali.citation_recommendation.citation

for entry in tqdm(paper_buaa.find()):
    paper_ali.save(entry)
for entry in tqdm(citation_buaa.find()):
    citation_ali.save(entry)
