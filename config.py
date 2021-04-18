import yaml
from util import *

with open("./config.yaml", "r", encoding="utf-8") as _fp:
    _yaml_obj = yaml.safe_load(_fp)

batch_size = _yaml_obj["batch_size"]
alpha = to_gpu(_yaml_obj["alpha"])
epoch = _yaml_obj["epoch"]
corpus_size = _yaml_obj["corpus_size"]
lr = _yaml_obj["lr"]
L2_penalty = float(_yaml_obj["L2_penalty"])
max_token_len = _yaml_obj["max_token_len"]
use_db_cache = _yaml_obj["use_db_cache"]
assert isinstance(use_db_cache, bool)
dataset_path = _yaml_obj["dataset_path"]
lookup_table_path = _yaml_obj["lookup_table_path"]
train_test_split_ratio = _yaml_obj["train_test_split_ratio"]
bert_hidden_size = _yaml_obj["bert_hidden_size"]
