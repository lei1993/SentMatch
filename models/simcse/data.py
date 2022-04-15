from torch.utils.data import DataLoader
from datasets import load_dataset
from collator import SimCSECollator


def load_data(args, tokenizer):
    """读取数据"""
    data_files = {"train": args.train_file}
    ds = load_dataset("text", data_files=data_files, cache_dir="./data")
    ds_tokenized = ds.map(lambda example: tokenizer(example["text"]), num_proc= args.num_proc)
    collator = SimCSECollator(tokenizer, max_len=args.max_length)
    dl = DataLoader(
        ds_tokenized["train"],
        batch_size=args.batch_size,
        collate_fn=collator.collate
    )
    return dl






