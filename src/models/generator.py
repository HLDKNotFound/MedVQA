import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

from .image_encoder import ImageEncoder
from .fusion_module import FusionModule

class MedVQAGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.cfg = config
        self.vision_encoder = ImageEncoder(
            config["model"]["vision_encoder"],
            freeze=config["model"].get("freeze_vision_encoder", True)
        )

        self.llm = AutoModelForCausalLM.from_pretrained(
            config["model"]["llm_model"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["llm_model"],
            trust_remote_code=True
        )

        self.fusion = FusionModule(
            vision_dim=self.vision_encoder.hidden_size,
            llm_dim=self.llm.config.hidden_size,
            num_hidden_layers=1
        )

        lora_cfg = LoraConfig(
            r=config["peft"]["lora_r"],
            lora_alpha=config["pert"]["lora_alpha"],
            target_modules=config["peft"]["target_modules"],
            lora_dropout=config["peft"]["lora_dropout"],
            bias="none",
            task_type="CAUSL_LM"
        )
        self.llm = get_peft_model(self.llm, lora_cfg)
        self.llm.print_trainable_parameters()

    def _build_inputs(self, images, questions, answers=None):
        pixel_values = self.vision_encoder.preprocess(images).to(self.llm.device)
        _, pooled = self.vision_encoder(pixel_values)
        image_embeds = self.fusion(pooled).unsqueeze(1) # [B, 1, llm_dim]

        if answers is not None:
            texts = [f"Question: {q}\nAnswer: {a}{self.tokenizer.eos_token}"\
                    for q, a in zip(questions, answers)]
            
        else:
            texts = [f"Question: {q}\nAnswer:"\
                    for q in questions]
            
        tok = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True,
            truncation=True
        ).to(self.llm_device)

        input_ids = tok.input_ids
        attention_mask = tok.attention_mask

        text_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        image_mask = torch.ones((image_embeds.size(0), 1), dtype=torch.long, device=input_ids.device)
        attention_mask = torch.cat([image_mask, attention_mask], dim=1)

        labels = None
        if answers is not None:
            labels = input_ids.clone()
            prompt_texts = [f"Question: {q}\nAnswer:" for q in questions]
            prompt_tok = self.tokenizer(prompt_texts, add_special_tokens=False).input_ids
            
            for i, p_ids in enumerate(prompt_tok):
                prompt_len = len(p_ids) + 1 # +1 for image token
                labels[i, :prompt_len] = -100

            dummy = torch.full((labels.size(0), 1), -100, dtype=torch.long, device=labels.device)
            labels = torch.cat([dummy, labels], dim=1)

        return {
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_ids": input_ids
        }
    
    def forward(self, images, questions, answers=None):
        ins = self._build_inputs(images, questions, answers)

        return self.llm(
            inputs_embeds=ins["inputs_embeds"],
            attention_maks=ins["attention_mask"],
            labels=ins["labels"],
            return_dict=True
        )
    
    def forward_sequence(self, images, input_ids, attention_mask):
        "Used by GRPO to compute logits over generated token IDs."
        return None