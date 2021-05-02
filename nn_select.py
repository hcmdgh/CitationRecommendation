import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, models, SentencesDataset, losses

import config
import dataset
from model import *


embedding_model = models.Transformer('bert-base-uncased', max_seq_length=512)

tokens = [config.sep_token]
embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
embedding_model.auto_model.resize_token_embeddings(len(embedding_model.tokenizer))

pooling_model = models.Pooling(embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[embedding_model, pooling_model], device=device)


def train_and_eval():
    train_set, val_set, test_set = dataset.get_nn_select_datasets()
    train_dataset = SentencesDataset(train_set, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)
    train_loss = losses.TripletLoss(model=model)

    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(val_set, name='val_set')

    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=config.epoch,
              evaluation_steps=1000,
              warmup_steps=math.ceil(len(train_dataloader) * config.epoch * 0.1),
              output_path=config.model_state_path)


if __name__ == '__main__':
    train_and_eval()
