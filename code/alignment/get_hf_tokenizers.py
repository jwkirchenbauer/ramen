from transformers import AutoModel, AutoTokenizer
import sys
import os
import argparse


parser = argparse.ArgumentParser(
    description="Estimate word translation probability from aligned fastText vectors")

parser.add_argument('--src_name', required=True, help="shorthand name for src language/tokenizer")
parser.add_argument('--tgt_name', required=True, help="shorthand name for tgt language/tokenizer")
parser.add_argument('--src_hf_name', default=None, help="name or path of corresponding src tokenizer loadable by hf, defaults to src_name")
parser.add_argument('--tgt_hf_name', default=None, help="name or path of corresponding tgt tokenizer loadable by hf, defaults to tgt_name")
parser.add_argument('--base_save_path', required=True, help="base path to save the tokenizer vocabularies to")


if __name__ == "__main__":
    params = parser.parse_args()

    src_name = params.src_name
    tgt_name = params.tgt_name

    src_hf_name = params.src_hf_name if params.src_hf_name is not None else src_name
    tgt_hf_name = params.tgt_hf_name if params.tgt_hf_name is not None else tgt_name
    
    base_save_path = params.base_save_path

    # src_name = "bert-base-uncased"
    # tgt_name = "flaubert"

    # src_hf_name = "bert-base-uncased"
    # tgt_hf_name = "flaubert/flaubert_base_cased"

    print(f"Base save path: {base_save_path}")

    src_tokenizer_save_path = f"{base_save_path}/{src_name}"
    tgt_tokenizer_save_path = f"{base_save_path}/{tgt_name}"

    print(f"src save path: {src_tokenizer_save_path}")
    print(f"tgt save path: {tgt_tokenizer_save_path}")

    

    os.makedirs(src_tokenizer_save_path, exist_ok=True)
    os.makedirs(tgt_tokenizer_save_path, exist_ok=True)
    
    try:
        src_tokenizer = AutoTokenizer.from_pretrained(src_hf_name, use_fast=True)
    except Exception as e:
        print(f"Unable to load src tokenizer identified by {src_hf_name}.")
        raise e
    try:
        tgt_tokenizer = AutoTokenizer.from_pretrained(tgt_hf_name, use_fast=True)
    except Exception as e:
        print(f"Unable to load tgt tokenizer identified by {tgt_hf_name}.")
        raise e

    src_tokenizer.save_vocabulary(src_tokenizer_save_path,f"{src_name}")
    tgt_tokenizer.save_vocabulary(tgt_tokenizer_save_path,f"{tgt_name}")



