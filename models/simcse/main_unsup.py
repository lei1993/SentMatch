import os
import math
import logging
import argparse
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer

from data import load_data
from model import SimCSE
from losses import compute_SimCSE_loss
from eval_unsup import load_test_data, eval


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_file", type=str, default="../../data/simclue_public/corpus.txt",
                        help="train text file")
    parser.add_argument("--pretrained", type=str, default="hfl/chinese-bert-wwm-ext",
                        help="huggingface pretrained model")
    parser.add_argument("--model_out", type=str, default="./model/simclue_simcse_unsup", help="model output path")
    parser.add_argument("--num_proc", type=int, default=5, help="dataset process thread num")
    parser.add_argument("--max_length", type=int, default=100, help="sentence max length")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--epochs", type=int, default=200, help="epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--tao", type=float, default=0.5, help="temperature")
    parser.add_argument("--device", type=str, default="cpu", help="device")
    parser.add_argument("--display_interval", type=int, default=500, help="display interval")
    parser.add_argument("--save_interval", type=int, default=100, help="save interval")
    parser.add_argument("--pool_type", type=str, default="cls", help="pool type")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="dropout rate")
    parser.add_argument("--f_test", type=str, default="../../data/simclue_public/test_public.json", help="test file")
    parser.add_argument("--simcse_model_path", type=str, default="./model/simclue_simcse_unsup",
                        help="simcse trained model path")
    parser.add_argument("--log_dir", type=str, default="./logs/simclue_simcse_unsup", help="log dir")
    args = parser.parse_args()
    return args


def train(args):
    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained, mirror="tuna")

    logging.info("Evaluate Bert pretrained model")
    eval_device = torch.device("cuda:0")
    test_data = load_test_data(args.f_test, tokenizer, args.max_length)
    logging.info("test data {0}".format(len(test_data)))
    eval_pretrain_model = SimCSE(args.pretrained, "cls")
    bert_test_score = eval(test_data, eval_pretrain_model, device=eval_device, tokenizer=tokenizer)
    logging.info(u"bert model test score:{:.4f}".format(bert_test_score))
    torch.cuda.empty_cache()

    train_dl = load_data(args, tokenizer)
    model = SimCSE(args.pretrained, args.pool_type, args.dropout_rate).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model_out = Path(args.model_out)
    if not model_out.exists():
        os.makedirs(model_out)

    batch_idx = 0
    loss_min = math.inf

    logging.info(u"Start training")

    for epoch_idx in range(args.epochs):
        loss_epoch = 0
        model.train()
        for data in tqdm(train_dl):
            batch_idx += 1
            pred = model(input_ids=data["input_ids"].to(args.device),
                         attention_mask=data["attention_mask"].to(args.device),
                         token_type_ids=data["token_type_ids"].to(args.device)
                         )
            loss = compute_SimCSE_loss(pred, args.tao, args.device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()
            loss_epoch += loss
            if batch_idx % args.display_interval == 0:
                logging.info(f"batch_idx: {batch_idx}, loss: {loss:>10f}")
        log_writer.add_scalar("train_loss", loss_epoch / len(train_dl), epoch_idx)
        if loss_epoch / len(train_dl) < loss_min:
            loss_min = loss_epoch / len(train_dl)
            model.save(model_out)
            logging.info(f" model saved")

        logging.info("Eval on SimCSE")
        simcse_test_score = eval(test_data, model, device=args.device, tokenizer=tokenizer)
        logging.info(u"simcse model test score:{:.4f}".format(simcse_test_score))

        log_writer.add_scalar("bert pretrained model spearman correlation", bert_test_score, epoch_idx)
        log_writer.add_scalar("simclue_simcse_unsup", simcse_test_score, epoch_idx)


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    log_fmt = "%(asctime)s|%(name)s|%(levelname)s|%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
