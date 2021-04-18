import config
import torch

_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# _device = torch.device("cpu")


def clip_tokens(batch_encoding):
    input_ids = to_gpu(batch_encoding["input_ids"])
    token_type_ids = to_gpu(batch_encoding["token_type_ids"])
    attention_mask = to_gpu(batch_encoding["attention_mask"])
    max_len = config.max_token_len
    input_ids = input_ids[:, :max_len]
    for item in input_ids:
        if item[-1] != 0:
            item[-1] = 102
    token_type_ids = token_type_ids[:, :max_len]
    attention_mask = attention_mask[:, :max_len]
    return dict(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)


def to_gpu(x):
    if isinstance(x, int) or isinstance(x, float):
        return torch.tensor(x, dtype=torch.float, device=_device)
    elif isinstance(x, torch.Tensor):
        if x.dtype == torch.double:
            x = x.float()
        return x.to(device=_device)
    return x.to(device=_device)


_logging_fp = open("./output.log", "w", encoding="utf-8")


def print_plus(*args, **kwargs):
    print(*args, **kwargs, flush=True)
    print(*args, **kwargs, file=_logging_fp, flush=True)
