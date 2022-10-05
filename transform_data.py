import numpy
from tqdm import tqdm
import json
import argparse
import os

def transform(item: dict) -> dict:
    # Do some transformation
    result = {
        'id': item['Id'],
        'source': item['original'],
        'target': item['reference'],
        "candidates": [
            {
                "text": hypo['content'],
                "score": {
                    "rouge1": hypo['metrics']['rouge1'],
                    "rouge2": hypo['metrics']['rouge2'],
                    "rougeL": hypo['metrics']['rougeL'],
                }
            }
            for hypo in item['hypotheses'].values()
        ]
    }
    return result

def main(args: argparse.Namespace) -> None:
    # Load data
    print('Loading data from {}'.format(args.data_path))
    with open(args.data_path, 'r') as f:
        data = [json.loads(line) for line in f]

    transformed_data = []
    for item in data:
        transformed_data.append(transform(item))

    # Save data
    if args.output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", os.path.basename(args.data_path))
    else:
        output_path = args.output_path
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    with open(output_path, 'w') as f:
        for item in transformed_data:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()
    main(args)
