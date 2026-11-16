"""Encoder, projector and LoRA-tuned Vicuna, with a configurable number of vision tokens."""

import json
from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from cxr_vlm.data import IMAGE_TOKEN
from cxr_vlm.encoder import EMBED_DIM, NUM_PATCHES, VisionEncoder

LLM_DIM = 4096
LLM_DTYPE = torch.bfloat16
IGNORE_INDEX = -100


def build_projector():
    return nn.Sequential(nn.Linear(EMBED_DIM, LLM_DIM), nn.GELU(), nn.Linear(LLM_DIM, LLM_DIM))


def pool_tokens(features, num_vision_tokens, sample_tokens=None, generator=None):
    """Mean pool to one token, optionally over a random per-image subset of patches."""
    if num_vision_tokens == NUM_PATCHES:
        if sample_tokens is not None:
            raise ValueError("Token sampling applies to the one-token model only.")
        return features
    if sample_tokens is None or sample_tokens == features.shape[1]:
        return features.mean(dim=1, keepdim=True)
    if not 1 <= sample_tokens < features.shape[1]:
        raise ValueError(f"sample_tokens must be in [1, {features.shape[1]}], got {sample_tokens}")
    scores = torch.rand(features.shape[:2], generator=generator, device=features.device)
    keep = scores.argsort(dim=1)[:, :sample_tokens]
    kept = features.gather(1, keep.unsqueeze(-1).expand(-1, -1, features.shape[-1]))
    return kept.mean(dim=1, keepdim=True)


class ReportGenerator(nn.Module):
    def __init__(self, num_vision_tokens, encoder_checkpoint, llm_name, max_text_length=2048, lora=None, projector_path=None):
        super().__init__()
        if num_vision_tokens not in (0, 1, NUM_PATCHES):
            raise ValueError("num_vision_tokens must be 0 (no image), 1 or 1369.")
        self.num_vision_tokens = num_vision_tokens
        self.max_text_length = max_text_length
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=LLM_DTYPE)
        self.llm.gradient_checkpointing_enable()
        self.llm.config.use_cache = False
        if lora:
            targets = sorted({name.split(".")[-1] for name, module in self.llm.named_modules()
                              if isinstance(module, nn.Linear) and "lm_head" not in name})
            self.llm = get_peft_model(self.llm, LoraConfig(
                r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                target_modules=targets, bias="none", task_type=TaskType.CAUSAL_LM,
            ))
        if num_vision_tokens > 0:
            self.encoder = VisionEncoder(encoder_checkpoint)
            self.projector = build_projector().to(dtype=LLM_DTYPE)
            if projector_path:
                self.projector.load_state_dict(torch.load(projector_path, map_location="cpu"))

    def train(self, mode=True):
        super().train(mode)
        if self.num_vision_tokens > 0:
            self.encoder.eval()
        return self

    def _embed(self, ids):
        return self.llm.get_input_embeddings()(ids)

    def _device(self):
        return next(self.llm.parameters()).device

    def image_embeds(self, images, sample_tokens=None, generator=None):
        if self.num_vision_tokens == 0:
            return None
        device = next(self.encoder.parameters()).device
        features = self.encoder(images.to(device, non_blocking=True))
        tokens = pool_tokens(features, self.num_vision_tokens, sample_tokens, generator)
        return self.projector(tokens.to(dtype=next(self.projector.parameters()).dtype))

    def _tokenize(self, text):
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def build_chat_batch(self, prompts, image_embeds, inference=False):
        """Embed each prompt, splice its vision tokens at <image>, mask the prompt and pad."""
        device = self._device()
        sequences, labels = [], []
        for i, prompt in enumerate(prompts):
            response = prompt.split("ASSISTANT: ")[1] if "ASSISTANT: " in prompt else ""
            response_ids = self._tokenize(response) if response else []
            if image_embeds is None:
                ids = torch.tensor(self.tokenizer(prompt, add_special_tokens=True).input_ids, device=device)
                embeds = self._embed(ids)
            else:
                left, right = prompt.split(IMAGE_TOKEN)
                left_ids = torch.tensor([self.tokenizer.bos_token_id] + self._tokenize(left), device=device)
                ids = torch.tensor(self._tokenize(right), device=device)
                embeds = torch.cat([self._embed(left_ids), image_embeds[i], self._embed(ids)])
            target = torch.full((embeds.shape[0],), IGNORE_INDEX, device=device)
            if response_ids:
                target[-len(response_ids):] = ids[-len(response_ids):]
            sequences.append(embeds)
            labels.append(target)

        pad_id = self.tokenizer.eos_token_id if inference else 0
        pad_embed = self._embed(torch.tensor([pad_id], device=device))
        longest = max(seq.shape[0] for seq in sequences)
        padded, masks, padded_labels = [], [], []
        for seq, target in zip(sequences, labels):
            pad = longest - seq.shape[0]
            filler = pad_embed.expand(pad, -1)
            ones, zeros = torch.ones(seq.shape[0], device=device), torch.zeros(pad, device=device)
            ignore = torch.full((pad,), IGNORE_INDEX, device=device)
            if inference:  # left padding for generation
                padded.append(torch.cat([filler, seq]))
                masks.append(torch.cat([zeros, ones]))
                padded_labels.append(torch.cat([ignore, target]))
            else:
                padded.append(torch.cat([seq, filler]))
                masks.append(torch.cat([ones, zeros]))
                padded_labels.append(torch.cat([target, ignore]))
        return torch.stack(padded), torch.stack(masks).long(), torch.stack(padded_labels)

    def build_alignment_batch(self, findings, image_embeds):
        """Stage 1: BOS, vision tokens, then the findings, with no prompt."""
        device = self._device()
        tokenized = self.tokenizer(
            [f"{text}{self.tokenizer.eos_token}" for text in findings],
            padding="longest", truncation=True, max_length=self.max_text_length, return_tensors="pt",
        ).to(device)
        ids, mask = tokenized.input_ids, tokenized.attention_mask
        batch, num_tokens = ids.shape[0], image_embeds.shape[1]
        text = self._embed(ids)
        embeds = torch.cat([text[:, :1], image_embeds, text[:, 1:]], dim=1)
        attention = torch.cat([mask[:, :1], torch.ones(batch, num_tokens, device=device, dtype=mask.dtype), mask[:, 1:]], dim=1)
        labels = ids.masked_fill(mask == 0, IGNORE_INDEX).masked_fill(ids == self.tokenizer.bos_token_id, IGNORE_INDEX)
        labels = torch.cat([labels[:, :1], torch.full((batch, num_tokens), IGNORE_INDEX, device=device), labels[:, 1:]], dim=1)
        return embeds, attention, labels

    def forward(self, batch, stage):
        image_embeds = self.image_embeds(batch["image"])
        if stage == "alignment":
            embeds, mask, labels = self.build_alignment_batch(batch["findings"], image_embeds)
        elif stage == "finetune":
            embeds, mask, labels = self.build_chat_batch(batch["prompt"], image_embeds)
        else:
            raise ValueError(f"Unknown stage: {stage}")
        return self.llm(inputs_embeds=embeds, attention_mask=mask, labels=labels).loss

    @torch.no_grad()
    def generate_reports(self, batch, max_new_tokens=512, sample_tokens=None, generator=None, ablate_projector=False):
        image_embeds = self.image_embeds(batch["image"], sample_tokens, generator)
        if ablate_projector:  # Probe: the LLM still receives vision tokens, but they carry nothing.
            image_embeds = torch.zeros_like(image_embeds)
        embeds, mask, _ = self.build_chat_batch(batch["prompt"], image_embeds, inference=True)
        output = self.llm.generate(
            inputs_embeds=embeds, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id, use_cache=True,
        )
        return [text.strip() for text in self.tokenizer.batch_decode(output, skip_special_tokens=True)]

    def save(self, run_dir, metadata):
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        if self.num_vision_tokens > 0:
            torch.save(self.projector.state_dict(), run_dir / "projector.pt")
        if isinstance(self.llm, PeftModel):  # Stage 1 leaves the LLM frozen; only the LoRA adapter is worth saving.
            self.llm.save_pretrained(run_dir)
        (run_dir / "run.json").write_text(json.dumps(metadata, indent=2))


def load_trained(run_dir, encoder_checkpoint):
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "run.json").read_text())
    model = ReportGenerator(metadata["num_vision_tokens"], encoder_checkpoint, metadata["llm"])
    if metadata["num_vision_tokens"] > 0:
        model.projector.load_state_dict(torch.load(run_dir / "projector.pt", map_location="cpu"))
    model.llm = PeftModel.from_pretrained(model.llm, run_dir, is_trainable=False, torch_dtype=LLM_DTYPE)
    model.llm.gradient_checkpointing_disable()
    model.llm.config.use_cache = True
    return model.to("cuda").eval(), metadata
