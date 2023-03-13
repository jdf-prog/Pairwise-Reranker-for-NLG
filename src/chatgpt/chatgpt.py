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
    save_pkl_cand_scores,
)
from common.utils import (
    seed_everything,
    str2bool,
    empty2None
)
from common.evaluation import (
    overall_eval
)

def request_chatgpt(prompt, retry_limit=10, **kwargs):
    retried = 0
    while True:
        # try to generate candidates
        try:
            response = openai.ChatCompletion.create(
                messages=[{"role": "assistant", "content": prompt}],
                **kwargs)
            break
        except Exception as e:
            if retried < retry_limit:
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
                retried += 1
            else:
                print("Too many retries")
                raise e
    return response

def top_p_chatgpt(source, args):
    prompt = args.prefix + source
    config = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "frequency_penalty": args.frequency_penalty,
        "presence_penalty": args.presence_penalty,
        "n": args.n,
        "stop": args.stop,
        "api_key": args.api_key,
    }
    response = request_chatgpt(prompt, **config)
    candidates = [
        choice['message']['content'].strip("\n ")
        for choice in response['choices']]
    return candidates, response

def num_in_prompt_chatgpt(source, args):
    prompt = args.prefix + source
    config = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "frequency_penalty": args.frequency_penalty,
        "presence_penalty": args.presence_penalty,
        "n": 1,
        "stop": args.stop,
        "api_key": args.api_key,
    }
    retried = 0
    retry_limit = 5
    while True:
        response = request_chatgpt(prompt, **config)
        candidates = []
        content = response['choices'][0]['message']['content'].strip("\n ")
        contents = [c.strip() for c in content.split("\n")] # remove empty lines
        contents = [c for c in contents if c != ""] # remove non-numbered lines
        if len(contents) == args.n:
            break
        elif len(contents) < args.n:
            if retried < retry_limit:
                print("Too few candidates")
                print("Retrying...")
                retried += 1
                continue
            else:
                print("Too many retries")
                raise ValueError("Too few candidates")
        else:
            contents = contents[:args.n]
            break

    for text in contents[:args.n]:
        text = text.strip("\n ")
        for i, s in enumerate(text):
            if s.isalpha():
                break
        text = text[i:]
        candidates.append(text)
    return candidates, response

def main(args):
    ids, sources, targets = load_raw_dataset(args.dataset, args.set, load_shuffle=args.load_shuffle)
    print("Number of samples: {}".format(len(sources)))
    if args.load_shuffle:
        permu_idxs = np.random.permutation(len(sources))
        ids = [ids[i] for i in permu_idxs]
        sources = [sources[i] for i in permu_idxs]
        targets = [targets[i] for i in permu_idxs]

    # # align with gpt3 datasets
    # import torch
    # sources = torch.load(f"../../data/{args.dataset}_gpt3/{args.set}/sources.pkl")
    # targets = torch.load(f"../../data/{args.dataset}_gpt3/{args.set}/targets.pkl")
    if args.end_idx is not None:
        sources = sources[:args.end_idx]
        targets = targets[:args.end_idx]
    if args.start_idx is not None:
        sources = sources[args.start_idx:]
        targets = targets[args.start_idx:]
    if isinstance(args.max_size, int) and args.max_size > 0:
        sources = sources[:args.max_size]
        targets = targets[:args.max_size]

    save_pkl_sources_and_targets(
        args.save_dataset_name, args.set,
        sources,
        targets,
        start_idx=0,
        end_idx=args.max_size
    )
    exit()

    prompt_usage_tokens = []
    completion_usage_tokens = []
    candidates = []
    start_idx = args.start_idx
    end_idx = args.end_idx
    try:
        for i, (source, target) in tqdm(
            enumerate(zip(sources, targets)),
            desc="Generating with API",
            total=len(sources)
        ):
            if args.generation_method == "top_p_sampling":
                _candidates, response = top_p_chatgpt(source, args)
            elif args.generation_method == "num_in_prompt":
                _candidates, response = num_in_prompt_chatgpt(source, args)
            else:
                raise ValueError("Invalid generation method")
            candidates.append(_candidates)
            prompt_usage_tokens.append(response['usage']['prompt_tokens'])
            completion_usage_tokens.append(response['usage']['completion_tokens'])
    except Exception as e:
        print("Error while generating candidates with API")
        print("Saving...")
        if start_idx is not None:
            end_idx = start_idx + len(candidates)
        else:
            start_idx = 0
            end_idx = len(candidates)
        raise e
    finally:
        if isinstance(args.start_idx, int) and args.end_idx is None:
            end_idx = args.start_idx + len(candidates)
        save_pkl_sources_and_targets(
            args.save_dataset_name, args.set,
            sources,
            targets,
            start_idx=start_idx,
            end_idx=end_idx
        )
        save_pkl_candidates(
            args.save_dataset_name, args.set,
            generation_method=args.generation_method,
            model_name=args.model,
            candidates=candidates,
            start_idx=start_idx,
            end_idx=end_idx
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

    # evaluation
    if args.eval_metrics is not None:
        metrics = args.eval_metrics.split(",")
        scores = overall_eval(candidates, targets, metrics=metrics)
        for metric, score in scores.items():
            print(f"{metric.upper()}")
            print("Top Beam: {}".format(np.mean([s[0] for s in score])))
            print("Oracle Min: {}".format(np.mean([min(s) for s in score])))
            print("Oracle Max: {}".format(np.mean([max(s) for s in score])))
            print("Oracle Mean: {}".format(np.mean([np.mean(s) for s in score])))
            if args.save_scores:
                save_pkl_cand_scores(
                    args.save_dataset_name, args.set,
                    args.generation_method,
                    args.model,
                    metric,
                    score
                )


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
    parser.add_argument("--generation_method", type=str, default="top_p_sampling",
        choices=["top_p_sampling", "num_in_prompt"])

    # data config
    parser.add_argument("--dataset", type=str, default="cnndm",
        choices=["cnndm", "commongen", "wmt18"])
    parser.add_argument("--set", type=str, default="test",)
    parser.add_argument("--max_size", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--load_shuffle", type=str2bool, default=False)
    parser.add_argument("--eval_metrics", type=empty2None, default=None, help="split by comma")
    parser.add_argument("--save_scores", type=str2bool, default=False)

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
    save_dataset_names = ["cnndm_chatgpt", "commongen_chatgpt", "wmt18_chatgpt"]
    args.save_dataset_name = save_dataset_names[args.dataset_idx]
    prefixs = [
        "Summarize the following news article: \n",
        "Generate a sentence with the following words: \n",
        "Translate Chinese to English: \n"
    ]
    num_in_prompt_prefixs = [
        "Generate {} summary for the following article: \n".format(args.n),
        "Generate {} sentence with the following words. \n".format(args.n),
        "Translate Chinese to English. Provide {} different translations: \n".format(args.n)
    ]
    if args.generation_method == "num_in_prompt":
        args.prefix = num_in_prompt_prefixs[args.dataset_idx]
    elif args.generation_method == "top_p_sampling":
        args.prefix = prefixs[args.dataset_idx]
    else:
        raise ValueError("Unknown generation method: {}".format(args.generation_method))

    print(args)
    seed_everything(args.seed)
    main(args)
