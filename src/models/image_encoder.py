import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor

class ImageEncoder(nn.Module):
    def __init__(self, model_name="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", freeze=True):
        super().__init__()

        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.processor = CLIPImageProcessor.from_pretrained(model_name)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

        self.hidden_size = self.model.config.hidden_size

    def preprocess(self, images):
        # images: list of PIL.Image
        return self.processor(images, return_tensors="pt")["pixel_values"]
    
    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state, outputs.pooler_output
