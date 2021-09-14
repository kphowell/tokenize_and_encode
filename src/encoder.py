import torch
from transformers import DistilBertModel

class DistilBertEncoder(torch.nn.Module):
    """
    DistilBERT Model. Uses the DisstilBertModel class from Hugging Face.
    Loads either a pretrained HF model or a model form a specified path.
    """
    def __init__(self, model_name_or_path: str, gpu_id: int = -1):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(model_name_or_path)

        device_str = (
            f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id != -1 else "cpu"
        )
        self.device = torch.device(device_str)
        self.encoder.to(self.device)
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, tokenized_messages):
        input_ids = tokenized_messages.get("input_ids")
        attention_mask = tokenized_messages.get("attention_mask")

        results = self.encoder(input_ids.to(self.device), attention_mask=attention_mask.to(self.device))[0]

        return results