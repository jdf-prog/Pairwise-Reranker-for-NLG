import torch
import torch.nn as nn

def build_tokenizer(args):
    tokenizer = None
    if args.model_type.startswith("t5"):
        print("\nUsing T5 tokenizer")
        from transformers import T5TokenizerFast
        tokenizer = T5TokenizerFast.from_pretrained(args.model, cache_dir=args.cache_dir)
    elif args.model_type.startswith("pegasus"):
        print("\nUsing Pegasus tokenizer")
        from transformers import PegasusTokenizer
        tokenizer = PegasusTokenizer.from_pretrained(args.model, cache_dir = args.cache_dir)
    elif args.model_type.startswith("bart"):
        print("\nUsing Bart tokenizer")
        from transformers import BartTokenizerFast
        tokenizer = BartTokenizerFast.from_pretrained(args.model, cache_dir = args.cache_dir)
    else:
        from transformers import AutoTokenizer
        print(f"\nUsing {args.model_type.upper()} tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    return tokenizer

def build_model(args):
    model = None
    if args.model_type.startswith("t5"):
        print("\nUsing T5 model")
        from transformers import T5ForConditionalGeneration
        model = T5ForConditionalGeneration.from_pretrained(args.model, cache_dir = args.cache_dir)
    elif args.model_type.startswith("pegasus"):
        print("\nUsing Pegasus model")
        from transformers import PegasusForConditionalGeneration
        model = PegasusForConditionalGeneration.from_pretrained(args.model, cache_dir = args.cache_dir)
    elif args.model_type.startswith("bart"):
        print("\nUsing Bart model")
        from transformers import BartForConditionalGeneration
        model = BartForConditionalGeneration.from_pretrained(args.model, cache_dir = args.cache_dir)
    else:
        print(f"\nUsing {args.model_type.upper()} model")
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model, cache_dir = args.cache_dir)

    return model

class FTModel(nn.Module):

    def __init__(self, pretrained_model, args):

        super(FTModel, self).__init__()

        self.pretrained_model = pretrained_model
        self.args = args

    def forward(self, input_ids, attention_mask, labels):

        output = self.pretrained_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True
        )

        loss_ce = output["loss"]
        loss_ce = torch.nan_to_num(loss_ce)
        outputs = {
            "loss": loss_ce,
            "loss_ce": loss_ce,
        }

        return outputs
