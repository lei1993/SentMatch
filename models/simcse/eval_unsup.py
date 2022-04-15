import json
import logging
import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from model import SimCSE
from tqdm import tqdm


def load_test_data(fname, tokenizer, max_length=100):
    lines = open(fname, "r").read().splitlines()
    sentences = []
    for line in lines:
        line = json.loads(line)
        sa, sb, s = line["sentence1"], line["sentence2"], line["label"]
        sentences.append([sa, sb, float(s)])
    return sentences


def eval(sentences, model, tokenizer, device, batch_size = 256, max_length=100):
    model.eval()
    model.to(device)
    a_embeddings = []
    b_embeddings = []
    scores = []
    with torch.no_grad():
        total_batch = len(sentences) // batch_size + (1 if len(sentences) % batch_size > 0 else 0)
        for batch_id in tqdm(range(total_batch)):
            a_token = tokenizer(
                [ sent[0] for sent in sentences[batch_id * batch_size:(batch_id + 1) * batch_size]],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            b_token = tokenizer(
                [sent[1] for sent in sentences[batch_id * batch_size:(batch_id + 1) * batch_size]],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            a_embed = model(a_token["input_ids"].to(device),
                            a_token["attention_mask"].to(device),
                            a_token["token_type_ids"].to(device))
            b_embed = model(b_token["input_ids"].to(device),
                            b_token["attention_mask"].to(device),
                            b_token["token_type_ids"].to(device))

            a_embeddings.append(a_embed)
            b_embeddings.append(b_embed)
            scores.append(sent[2] for sent in sentences[batch_id * batch_size:(batch_id + 1) * batch_size])
        a_embeddings = torch.cat(a_embeddings, 0)
        b_embeddings = torch.cat(b_embeddings, 0)
        sim_score = F.cosine_similarity(a_embeddings, b_embeddings).cpu().numpy()
        scores = np.array([sent[2] for sent in sentences]).reshape(-1)
        corr = scipy.stats.spearmanr(sim_score, scores ).correlation
    return corr

def main():
    pretrained_model_path = "hfl/chinese-bert-wwm-ext"
    simcse_model_path = "./model/simclue_simcse_unsup"
    f_test = "./data/simclue_public/test_public.json"

    logging.info("Load tokenizer")
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
    max_length = 100
    device = torch.device("cuda:0")
    test_data = load_test_data(f_test, tokenizer, max_length)
    logging.info("test data {0}".format(len(test_data)))

    logging.info("eval bert model")
    model = SimCSE(pretrained_model_path, "cls")
    bert_test_score = eval(test_data, model, device=device, tokenizer=tokenizer)

    logging.info("eval simcse mdodel")
    model = SimCSE(simcse_model_path, "cls")
    simcse_test_score = eval(test_data, model, device=device, tokenizer=tokenizer)


    logging.info(u"bert model test score:{:.4f}".format(bert_test_score))
    logging.info(u"simcse model test score:{:.4f}".format(simcse_test_score))



if __name__ == '__main__':
    log_fmt = "%(asctime)s|%(name)s|%(levelname)s|%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
