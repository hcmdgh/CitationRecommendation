import yaml

with open("./config.yaml", "r", encoding="utf-8") as _fp:
    _yaml_obj = yaml.safe_load(_fp)

batch_size = _yaml_obj["batch_size"]
assert isinstance(batch_size, int)

alpha = _yaml_obj["alpha"]
epoch = _yaml_obj["epoch"]
lr = _yaml_obj["lr"]
L2_penalty = float(_yaml_obj["L2_penalty"])
max_token_len = _yaml_obj["max_token_len"]
use_db_cache = _yaml_obj["use_db_cache"]
assert isinstance(use_db_cache, bool)
nn_select_dataset_path = _yaml_obj["nn_select_dataset_path"]

nn_select_dataset_size = _yaml_obj["nn_select_dataset_size"]
assert isinstance(nn_select_dataset_size, int)

test_set_ratio = _yaml_obj["test_set_ratio"]
assert isinstance(test_set_ratio, float)

val_set_ratio = _yaml_obj["val_set_ratio"]
assert isinstance(val_set_ratio, float)

bert_hidden_size = _yaml_obj["bert_hidden_size"]
use_gpu = _yaml_obj["use_gpu"]
assert isinstance(use_gpu, bool)
model_state_path = _yaml_obj["model_state_path"]

sep_token = _yaml_obj["sep_token"]
assert isinstance(sep_token, str)
