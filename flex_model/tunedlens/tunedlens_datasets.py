import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from llama import Tokenizer


# TODO: WIP
def main():
    tokenizer = AutoTokenizer.from_pretrained(
        "/ssd005/projects/llm/llama-2-13b-hf",
        local_files_only=True,
    )
    tokenizer.pad_token_id = 0
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = "right"

    def tokenize(ps):
        return tokenizer(ps, padding=True, return_tensors="pt")

    dataset = load_dataset("wikitext", "wikitext-103-v1")

    original_tokenizer = Tokenizer("/ssd005/projects/llm/llama-2-13b/tokenizer.model")

    breakpoint()


if __name__ == "__main__":
    main()
