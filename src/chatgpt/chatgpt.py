import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import openai
import argparse
import time
import numpy as np
from tqdm import tqdm
from openai.error import RateLimitError, Timeout
from transformers import RobertaTokenizer
from common.data import (
    load_raw_dataset,
    save_pkl_candidates,
    save_pkl_sources_and_targets,
)
from common.utils import (
    seed_everything
)

def main(args):
    ids, sources, targets = load_raw_dataset(args.dataset, args.set)
    idxs = np.arange(len(sources))
    np.random.shuffle(idxs)
    ids = [ids[idx] for idx in idxs][:args.max_size]
    sources = [sources[idx] for idx in idxs][:args.max_size]
    targets = [targets[idx] for idx in idxs][:args.max_size]
    print("Number of samples: {}".format(len(sources)))

    # # align with gpt3 datasets
    # import torch
    # sources = torch.load(f"../../data/{args.dataset}_gpt3/{args.set}/sources.pkl")
    # targets = torch.load(f"../../data/{args.dataset}_gpt3/{args.set}/targets.pkl")
    # sources = sources[:args.max_size]
    # targets = targets[:args.max_size]

    sources = sources[args.start_idx:] if args.start_idx is not None else sources
    targets = targets[args.start_idx:] if args.start_idx is not None else targets

    prompt_usage_tokens = []
    completion_usage_tokens = []
    candidates = []
    try:
        for i, (source, target) in tqdm(
            enumerate(zip(sources, targets)),
            desc="Generating with API",
            total=len(sources)
        ):
            prompt = args.prefix + source
            retry_time = 0
            while True:

                # try to generate candidates
                try:
                    response = openai.ChatCompletion.create(
                        model=args.model,
                        messages=[{"role": "assistant", "content": prompt}],
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        frequency_penalty=args.frequency_penalty,
                        presence_penalty=args.presence_penalty,
                        n=args.n,
                        stop=args.stop,
                        api_key=args.api_key)
                    break
                except Exception as e:
                    if retry_time < 10:
                        # if rate limit error, wait 2 seconds and retry
                        if isinstance(e, RateLimitError):
                            print("Rate limit error")
                            print("Retrying...")
                            time.sleep(2)
                            continue
                        elif isinstance(e, Timeout):
                            print("Timeout error")
                            print("Retrying...")
                            time.sleep(2)
                            continue
                        retry_time += 1
                    else:
                        print("Too many retries")
                        raise e
            _candidates = [
                choice['message']['content'].strip("\n ")
                for choice in response['choices']]
            # print("prompt: ", prompt)
            # print("target: ", target)
            # print("candidates: ", _candidates)
            candidates.append(_candidates)
            prompt_usage_tokens.append(response['usage']['prompt_tokens'])
            completion_usage_tokens.append(response['usage']['completion_tokens'])
    except Exception as e:
        print("Error while generating candidates with API")
        print("Saving...")
        start_idx = args.start_idx if args.start_idx is not None else 0
        end_idx = start_idx + len(candidates)
        save_pkl_sources_and_targets(
            args.save_dataset_name, args.set,
            sources[start_idx:end_idx],
            targets[start_idx:end_idx],
            start_idx=start_idx,
            end_idx=end_idx
        )
        save_pkl_candidates(
            args.save_dataset_name, args.set,
            generation_method="top_p_sampling",
            model_name="chatgpt",
            candidates=candidates,
            start_idx=start_idx,
            end_idx=end_idx
        )
        raise e
    save_pkl_sources_and_targets(
        args.save_dataset_name, args.set,
        sources, targets)
    save_pkl_candidates(
        args.save_dataset_name, args.set,
        generation_method="top_p_sampling",
        model_name="chatgpt",
        candidates=candidates
    )

    print("Generate {} candidate per data point".format(args.n))
    print("{} data points in total".format(len(sources)))
    print("Average prompt usage tokens: {}".format(np.mean(prompt_usage_tokens)))
    print("Average completion usage tokens: {} ({:0.4f} tokens per candidate)".format(
        np.mean(completion_usage_tokens), np.mean(completion_usage_tokens) / args.n))
    total_tokens = (np.sum(completion_usage_tokens) + np.sum(prompt_usage_tokens))
    total_cost = total_tokens * 0.002 / 1000 # 0.002 USD per 1k tokens
    print("Total estimated cost (USD): ${:0.4f} (${:0.4f} per data point)".format(
        total_cost, total_cost / len(sources)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # GPT-3 config
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
        choices=["gpt-3.5-turbo"])
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1, help="Higher values means the model will take more risks., [0.0, 1.0]")
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--frequency_penalty", type=float, default=0)
    parser.add_argument("--presence_penalty", type=float, default=0)
    parser.add_argument("--n", type=int, default=10, help="How many completions to generate for each prompt.")
    parser.add_argument("--stop", type=str, default=None, \
        help="Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence.")

    # data config
    parser.add_argument("--dataset", type=str, default="cnndm",
        choices=["cnndm", "commongen", "wmt18"])
    parser.add_argument("--set", type=str, default="test",)
    parser.add_argument("--max_size", type=int, default=1000)
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)

    # others
    parser.add_argument("--seed", type=int, default=42)


    args = parser.parse_args()

    if args.api_key is None:
        args.api_key = os.environ.get("OPENAI_API_KEY")
        if args.api_key is None:
            raise ValueError("Please set your API key.")
    openai.api_key = args.api_key

    # set dataset specific config
    args.dataset_idx = ["cnndm", "commongen", "wmt18"].index(args.dataset)
    prefixs = [
        "Summarize the following news article: \n",
        "Generate a sentence with the following words: \n",
        "Translate Chinese to English: \n"
    ]
    save_dataset_names = ["cnndm_chatgpt", "commongen_chatgpt_", "wmt18_chatgpt"]
    max_tokens = [128, 30, 100] # set limits
    args.prefix = prefixs[args.dataset_idx]
    args.save_dataset_name = save_dataset_names[args.dataset_idx]
    args.max_tokens = max_tokens[args.dataset_idx] if args.max_tokens is None else args.max_tokens
    args.max_tokens = None # debug

    print(args)
    seed_everything(args.seed)
    main(args)
