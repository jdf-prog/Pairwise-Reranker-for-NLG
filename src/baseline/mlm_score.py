

import json
import argparse
import torch
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
    scores = []
    for i in tqdm(range(len(candidates))):
        scores.append(scorer.score_sentences(candidates[i]))
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
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--candidate_file', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--max_size', type=int, default=-1)
    args = parser.parse_args()
    print(args)

    main(args)
