

import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from mlm.scorers import MLMScorer, MLMScorerPT, LMScorer
from mlm.models import get_pretrained
import mxnet as mx
def main(args):
    if isinstance(args.n_gpu, int) and args.n_gpu > 0:
        print(f"Using {args.n_gpu} GPUs")
        ctxs = [mx.gpu(i) for i in range(args.n_gpu)]
    else:
        print("Using CPU")
        ctxs = [mx.cpu()]
    model, vocab, tokenizer = get_pretrained(ctxs, args.model)
    scorer = MLMScorer(model, vocab, tokenizer, ctxs)

    # read candidates
    candidates = torch.load(args.candidate_file)
    if isinstance(args.max_size, int) and args.max_size > 0:
        print(f"Limiting to {args.max_size} candidates")
        candidates = candidates[:args.max_size]

    cand_idxs = np.zeros((len(candidates), len(candidates[0])), dtype=np.int32)
    flatten_cands = []
    idx = 0
    for i in range(len(candidates)):
        for j in range(len(candidates[i])):
            flatten_cands.append(candidates[i][j])
            cand_idxs[i, j] = idx
            idx += 1

    flatten_scores = scorer.score_sentences(
        flatten_cands,
        split_size=args.split_size,
        num_workers=args.num_workers,
        ratio=args.ratio
        )

    scores = np.zeros((len(candidates), len(candidates[0])))
    for i in range(len(candidates)):
        for j in range(len(candidates[i])):
            scores[i, j] = flatten_scores[cand_idxs[i, j]]
    torch.save(scores, args.save_path)
    print("Saving mlm-scores to {}".format(args.save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='roberta-large-en-cased',
        choices = ['bert-base-en-uncased', 'bert-base-en-cased',
                    'roberta-base-en-cased', 'bert-large-en-uncased',
                    'bert-large-en-cased', 'roberta-large-en-cased',
                    'bert-base-en-uncased-owt', 'bert-base-multi-uncased',
                    'bert-base-multi-cased',
                    'gpt2-117m-en-cased', 'gpt2-345m-en-cased'
        ]
    )
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--candidate_file', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--max_size', type=int, default=-1)
    parser.add_argument('--split_size', type=int, default=40)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--ratio', type=float, default=0.1)
    parser.add_argument('--start')

    args = parser.parse_args()
    print(args)

    main(args)
