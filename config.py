import yaml
from util import *

with open("./config.yaml", "r", encoding="utf-8") as _fp:
    _yaml_obj = yaml.safe_load(_fp)

dataset_path = _yaml_obj["dataset_path"]
dataset_paper_cnt = _yaml_obj["dataset_paper_cnt"]
negative_samples_per_pair = _yaml_obj["negative_samples_per_pair"]
batch_size = _yaml_obj["batch_size"]
alpha = to_gpu(_yaml_obj["alpha"])
epoch = _yaml_obj["epoch"]
corpus_size = _yaml_obj["corpus_size"]
lr = _yaml_obj["lr"]
L2_penalty = float(_yaml_obj["L2_penalty"])
max_token_len = _yaml_obj["max_token_len"]
use_db_cache = _yaml_obj["use_db_cache"]
assert isinstance(use_db_cache, bool)
dataset_pkl_path = _yaml_obj["dataset_pkl_path"]
