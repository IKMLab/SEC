from typing import Tuple
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import trange
from pathlib import Path, PurePosixPath
import spacy
import json
import codecs
from t5_utils import load_filtering_model, t5_generate, sentence_filtering


def prepare_data(data_path: str, gen_model_name: str) -> Tuple[list, list]:
    # Load the dataset.
    # Format: one-sentence-per-line file.
    with open(data_path, "r") as f:
        data = f.read().split('\n')

    # Add the instruction for T5.
    if gen_model_name.startswith("t5"):
        t5_input_sents = ["summarize: " + d for d in data]
    else:
        t5_input_sents = data

    return data, t5_input_sents


def main(args):
    device = torch.device(
        f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    )
    do_sample = True if args.decoding_strategy in ['top-k', 'top-p'] else False

    if args.decoding_strategy == 'top-p':
        decoding_value = args.topp_value
    elif args.decoding_strategy == 'top-k':
        decoding_value = args.topk_value
    elif args.decoding_strategy == 'beam':
        decoding_value = args.beam_size

    data, sentences = prepare_data(args.data_path, args.gen_model_name)
    data_name = Path(args.data_path).stem

    if args.gen_model_name not in ["t5-base", "t5-small", "t5-large", "eda"] \
            and not args.gen_model_name.startswith('t5-3b'):
        checkpoint = args.gen_model_name.split("/")[1]
    else:
        checkpoint = args.gen_model_name

    if args.gen_model_name == 'eda':
        save_seqs_name = f'eda_nlp/data/{checkpoint}_{data_name}_' + \
            f'{args.num_generate_per_sentence}N.txt'
    else:
        # E.g. t5-3b_huffpost_10N_top-k_40.txt
        save_seqs_name = f'experiments/{checkpoint}_{data_name}_' + \
            f'{args.num_generate_per_sentence}N_' + \
            f'{args.decoding_strategy}_{decoding_value}_l{args.output_length}.txt'

    if Path(save_seqs_name).exists():
        print(f"{save_seqs_name} exists!!")
        if args.gen_model_name == 't5-3b':
            with open(save_seqs_name, "r", encoding='unicode_escape') as f:
                generated_sentences = [
                    line[2:-1] for line in codecs.decode(f.read().encode('raw_unicode_escape')).split('\n')
                ]
        elif args.gen_model_name == 'eda':
            df = pd.read_csv(save_seqs_name, sep="\t", header=None)
            generated_sentences = df.iloc[:, 1].tolist()
        else:
            with open(save_seqs_name, "r") as f:
                generated_sentences = f.read().splitlines()

    else:
        generated_sentences = t5_generate(
            ori_sentences=sentences,
            batch_size=args.batch_size,
            gen_model_name=args.gen_model_name,
            num_generate_per_sentence=args.num_generate_per_sentence,
            decoding_strategy=args.decoding_strategy,
            decoding_value=decoding_value,
            output_length=args.output_length,
            device=device,
            save_seqs_name=save_seqs_name,
        )

    if args.num_generate_per_sentence == 1:
        # N 為 1 的時候直接跳過過濾
        print("Directly concatenate sentences without filtering.")
        best_sentences = generated_sentences
        new_json_name = Path('data').joinpath(
            PurePosixPath(save_seqs_name).stem + '.json'
        )

    elif args.num_generate_per_sentence > 1:
        print("Start filter sentences...")
        print("Start appending prompts...")

        # Load the Filtering model.
        NLI_tokenizer, NLI_model = load_filtering_model(
            model_name=args.cls_model_name,
            device=device
        )
        NLI_model.eval()

        not_passed_counts = 0
        best_sentences = []
        pbar = trange(len(data))
        for k in pbar:
            ori_sentence = data[k]
            ori_sen_batch = [ori_sentence] * args.num_generate_per_sentence
            # Use list comprehension to append new_sen_batch.
            if args.gen_model_name == 'eda':
                new_sen_batch = generated_sentences[k * (args.num_generate_per_sentence+1):(
                    k+1)*(args.num_generate_per_sentence+1)-1]
            else:
                new_sen_batch = [generated_sentences[k * args.num_generate_per_sentence + i]
                                 for i in range(args.num_generate_per_sentence)]

            best_choice, got_passed = sentence_filtering(
                origin_sentence=ori_sen_batch,
                new_sentence=new_sen_batch,
                tokenizer=NLI_tokenizer,
                model=NLI_model,
                filter_mode=args.filter_mode,
                device=device
            )
            if not got_passed:
                not_passed_counts += 1

            pbar.set_postfix(not_passed=not_passed_counts)

            if best_choice != -1:
                best_sent = new_sen_batch[best_choice]
            else:
                best_sent = ""
            best_sentences.append(best_sent)

        report = f"The {args.num_generate_per_sentence} generated sentences " + \
                f"from {not_passed_counts} sentences " + \
                f"aren't predicted as entailment."
        print(report)
        print(f"The pass rate of {args.filter_mode} is {(len(data)-not_passed_counts)/len(data)}")

        new_json_name = Path('data').joinpath(
            PurePosixPath(save_seqs_name).stem +
            f'_{args.cls_model_name}_{args.filter_mode}.json'
        )

    assert len(best_sentences) == len(data)

    with open(new_json_name, "w") as f:
    # Concatenate and save files
        if args.output_tokens:
            nlp = spacy.load('en_core_web_sm')

            for i in range(len(data)):
                ori_tokens = [token.text for token in nlp(data[i])]
                new_tokens = [token.text for token in nlp(best_sentences[i])]
                concatenated = ori_tokens + new_tokens
                row = {'text': concatenated}
                # Save one entry in a json format.
                json.dump(row, f)
                f.write("\n")

        else:
            for i in range(len(data)):
                concatenated = data[i] + " " + best_sentences[i]
                row = {'text': concatenated}
                json.dump(row, f)
                f.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu_id",
        default=1,
        type=int
    )
    parser.add_argument(
        "--data_path",
        type=str,
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int
    )
    parser.add_argument(
        "--gen_model_name",
        default="t5-large",
    )
    parser.add_argument(
        "--cls_model_name",
        default="roberta-large-mnli",
    )
    parser.add_argument(
        "--decoding_strategy",
        default='top-k',
        choices=['greedy', 'top-k', 'top-p', 'beam']
    )
    parser.add_argument(
        "--topp_value",
        default=0.9,
        type=float
    )
    parser.add_argument(
        "--topk_value",
        default=40,
        type=int
    )
    parser.add_argument(
        "--beam_size",
        default=5,
        type=int
    )
    parser.add_argument(
        "--num_generate_per_sentence",
        default=10,
        type=int,
        help="一開始總共產生幾個句子",
    )
    parser.add_argument(
        "--filter_mode",
        default='E_only',
        choices=['E_only', 'EorN', 'C_only', 'N_only']
    )
    parser.add_argument(
        "--output_length",
        default='128',
        type=int,
    )
    parser.add_argument(
        '--output_tokens',
        action="store_true"
    )
    args = parser.parse_args()
    main(args)
