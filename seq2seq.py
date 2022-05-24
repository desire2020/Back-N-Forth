# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import json
import torch
import nltk
import nltk.tokenize
import torch.utils.data as data
import yake
import numpy as np
from IPython import embed

import tqdm

class Seq2seqSequentialDataset(data.Dataset):
    def __init__(self, tokenizer):
        language = "en"
        max_ngram_size = 1
        deduplication_thresold = 0.9
        deduplication_algo = 'seqm'
        windowSize = 1
        numOfKeywords = 8
        self.tokenizer = tokenizer
        self.kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size,
                                                    dedupLim=deduplication_thresold,
                                                    dedupFunc=deduplication_algo,
                                                    windowsSize=windowSize, top=numOfKeywords,
                                                    features=None)
        self.max_len = 512
        self.capacity = 16
        self.sequence_buffer = torch.zeros(16, self.max_len, dtype=torch.long)
        self.inverse_sequence_buffer = torch.zeros(16, self.max_len, dtype=torch.long)
        self.length_buffer = torch.zeros(16, dtype=torch.long)
        self.inv_length_buffer = torch.zeros(16, dtype=torch.long)
        self.size = 0
        self.inversed_condition = torch.zeros(16, self.max_len, dtype=torch.long)

    def expand_capacity(self):
        self.sequence_buffer = torch.cat((self.sequence_buffer, torch.zeros_like(self.sequence_buffer)), dim=0)
        self.inverse_sequence_buffer = torch.cat((self.inverse_sequence_buffer, torch.zeros_like(self.inverse_sequence_buffer)), dim=0)
        self.length_buffer = torch.cat((self.length_buffer, torch.zeros_like(self.length_buffer)), dim=0)
        self.inv_length_buffer = torch.cat((self.inv_length_buffer, torch.zeros_like(self.inv_length_buffer)), dim=0)
        self.capacity *= 2

    def add(self, json_loc):
        self.tokenizer.bos_token = "$"
        self.tokenizer.sep_token = "#"
        json_file = json.load(open(json_loc, "r"))
        num_skipped = 0
        iterator = tqdm.tqdm(json_file)
        for i, instance in enumerate(iterator):
            if i == 0:
                continue
            if self.size == self.capacity:
                self.expand_capacity()
            title = instance["title"]
            story = "\n".join([sent for sent in nltk.tokenize.sent_tokenize(instance["text"]) if sent.lower().find("http") == -1])

            if story.find("deleted") != -1 or story.find("removed") != -1 or story.find(""):
                num_skipped += 1
                iterator.write("Skipped %d deleted ones." % num_skipped)
                continue
            keywords = self.kw_extractor.extract_keywords(story)
            keywords_ids = [(i, story.find(key)) for i, (key, importance) in enumerate(keywords)]
            dtype = [("word", int), ("loc", int)]
            keywords_sorted = np.sort(np.array(keywords_ids, dtype), order="loc")
            keywords_sorted = [keywords[key_id][0] for (key_id, loc) in keywords_sorted]
            ids = [self.tokenizer.bos_token_id] + self.tokenizer.encode(title) + [self.tokenizer.sep_token_id] + self.tokenizer.encode(story) + [self.tokenizer.eos_token_id]
            if len(ids) > self.max_len or len(ids) < 25:
                num_skipped += 1
                iterator.write("Skipped %d weird lengthed ones." % num_skipped)
                continue
            inv_ids = [self.tokenizer.bos_token_id] + self.tokenizer.encode(story) + [self.tokenizer.sep_token_id] + self.tokenizer.encode(title) + [self.tokenizer.eos_token_id]
            self.sequence_buffer[self.size, 0:len(ids)] = torch.tensor(ids)
            self.inverse_sequence_buffer[self.size, 0:len(ids)] = torch.tensor(ids)
            self.length_buffer[self.size] = len(ids) - 1
            self.inv_length_buffer[self.size] = len(inv_ids) - 1
            self.size += 1

    def __getitem__(self, item):
        mask = torch.zeros(self.max_len-1, dtype=torch.float)
        mask[0: self.length_buffer[item]] = 1.0
        inv_mask = torch.zeros(self.max_len-1, dtype=torch.float)
        inv_mask[0: self.inv_length_buffer[item]] = 1.0
        return self.sequence_buffer[item], self.inverse_sequence_buffer[item], mask, inv_mask, self.length_buffer[item], self.inv_length_buffer[item]

    def __len__(self):
        return self.size

import transformers
import os
import pickle
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import argparse
from transformers import AdamW

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_mode",
        default=False,
        type=str2bool,
        required=False,
        help="load the latest checkpoint and run the eval pipeline.",
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        required=False,
        help="effective batchsize",
    )
    parser.add_argument(
        "--iter_per",
        default=8,
        type=int,
        required=False,
        help="cumulative grad step interval",
    )
    args = parser.parse_args()
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    inv_model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.bos_token = "$"
    tokenizer.sep_token = "#"
    if not os.path.exists("./checkpoints"):
        os.mkdir("checkpoints")
    try:
        dataset = pickle.load(
            open("checkpoints/seq2seq-dataset", "rb"))
        # raise NotImplementedError()
    except:
        dataset = Seq2seqSequentialDataset(tokenizer)
        dataset.add("./data/scary_story.json")
        pickle.dump(dataset, open("checkpoints/seq2seq-dataset", "wb"))

    base_model.train()
    base_model.cuda(0)
    inv_model.train()
    inv_model.cuda(1)
    opt = AdamW(lr=2e-5, weight_decay=0.02,
                eps=1e-8, params=base_model.parameters())
    i_opt = AdamW(lr=2e-5, weight_decay=0.02,
                eps=1e-8, params=inv_model.parameters())

    dataloader = data.DataLoader(
        dataset, batch_size=args.batch_size // args.iter_per, shuffle=True, drop_last=False, pin_memory=True,
        num_workers=8
    )
    if args.eval_mode:
        base_model.load_state_dict(torch.load("checkpoints/pretrained-seq2seq"))
        base_model.eval()
        inv_model.load_state_dict(torch.load("checkpoints/inv_model-seq2seq"))
        inv_model.eval()
        line = input("Please input title for your scary story:")
        input_ids = torch.tensor([tokenizer.bos_token_id] + tokenizer.encode(tokenizer.encode("%s"% line.strip())) + [tokenizer.sep_token_id])
        generated = lambda: base_model.generate(input_ids=input_ids.cuda().unsqueeze(dim=0), max_length=500,
                                       do_sample=True, top_p=0.80, top_k=10, repetition_penalty=1.2, pad_token_id=tokenizer.eos_token_id)

        # generated = base_model.generate(input_ids=input_ids.cuda().unsqueeze(dim=0), max_length=500, beam_size=10, pad_token_id=tokenzier.eos_token_id)

        embed(); exit()

    for epoch_idx in range(20):
        iterator = tqdm.tqdm(dataloader)
        F_LOSS = []
        I_LOSS = []
        for iter_count, (forward_ids, backward_ids, mask, inv_mask, f_len, b_len) in enumerate(iterator):
            f_maxlen = f_len.max() + 1
            b_maxlen = b_len.max() + 1
            input_ids = forward_ids[:, 0:f_maxlen].cuda(0)
            mask = mask[:, 0:f_maxlen-1].cuda(0)
            logits = base_model(input_ids).logits.log_softmax(dim=-1)[:, :-1, :]
            nll_all = - (logits.gather(dim=-1, index=input_ids[:, 1:].unsqueeze(dim=-1)).reshape_as(mask) * mask)
            f_nll_reduced = nll_all.sum(dim=-1)

            input_ids = backward_ids[:, 0:b_maxlen].cuda(1)
            inv_mask = inv_mask[:, 0:b_maxlen-1].cuda(1)
            logits = inv_model(input_ids).logits.log_softmax(dim=-1)[:, :-1, :]
            nll_all = -(logits.gather(dim=-1, index=input_ids[:, 1:].unsqueeze(dim=-1)).reshape_as(inv_mask) * inv_mask)
            b_nll_reduced = nll_all.sum(dim=-1)

            f_nll_reduced = f_nll_reduced.mean(dim=0)
            b_nll_reduced = b_nll_reduced.mean(dim=0)
            if iter_count % args.iter_per == 0:
                opt.zero_grad()
                i_opt.zero_grad()

            (f_nll_reduced / args.iter_per).backward()
            (b_nll_reduced / args.iter_per).backward()
            F_LOSS.append(f_nll_reduced.cpu().item())
            I_LOSS.append(b_nll_reduced.cpu().item())

            if iter_count % args.iter_per == args.iter_per - 1:
                opt.step()
                i_opt.step()
                if (iter_count // args.iter_per) % 25 == 0:
                    iterator.write("Iteration %d-%d, F-Loss %f, B-Loss %f" % (
                    epoch_idx, iter_count // args.iter_per, np.mean(F_LOSS), np.mean(I_LOSS)))

        torch.save(base_model.state_dict(), "checkpoints/pretrained-seq2seq")
        torch.save(inv_model.state_dict(), "checkpoints/inv_model-seq2seq")
        torch.save(opt.state_dict(), "checkpoints/opt-seq2seq")





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
