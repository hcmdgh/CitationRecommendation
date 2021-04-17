import yaml

with open("./config.yaml", "r", encoding="utf-8") as _fp:
    _yaml_obj = yaml.safe_load(_fp)

dataset_path = _yaml_obj["dataset_path"]
dataset_paper_cnt = _yaml_obj["dataset_paper_cnt"]
negative_samples_per_pair = _yaml_obj["negative_samples_per_pair"]
batch_size = _yaml_obj["batch_size"]
