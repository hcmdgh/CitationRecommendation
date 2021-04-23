import torch
import torch.nn as nn

import db
from util import *
from model import *


def embed_paper(paper):
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    with torch.no_grad():
        embedding = model(title_batch=[title], abstract_batch=[abstract])
    embedding = embedding.numpy()
    paper["embedding"] = embedding[0].tolist()
    return paper


if __name__ == '__main__':
    model = to_gpu(NNSelect())
    model.load_state_dict(torch.load("./data/model_state/model_epoch_98.pt"))
    db.map_papers(embed_paper)
