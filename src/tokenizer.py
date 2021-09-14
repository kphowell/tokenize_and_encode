from transformers import DistilBertTokenizerFast
from typing import List

class DistilBertTokenizer:
    """
    Tokenizer for DistilBERT. Uses the DisstilBertTokenizerFast tokenizer from
    Hugging Face. Loads either a pretrained HF tokenizer or a tokenizer form a
    specified path.
    """

    def __init__(self, tokenizer_name_or_path: str):
        super().__init__()
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_name_or_path)


    def forward(self, list_of_messages: List[str]):
        output = self.tokenizer(
            list_of_messages,
            return_tensors="pt",
            truncation=True,
            max_length=100,
            padding=True,
        )
        return output