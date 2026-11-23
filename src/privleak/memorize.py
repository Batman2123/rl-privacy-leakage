"""Step 1 of the experiment: make Qwen memorize the fake employee directory.

We use LoRA: the 1.7B original weights stay frozen and we train small add-on
matrices (~1% of the parameters). This fits on a free Colab GPU and matches
the original paper, which also trained with LoRA.
"""
import math
import random

import torch
from peft import LoraConfig, get_peft_model

# Which layers get LoRA add-ons: the attention projections (q, k, v, o) and the
# MLP projections (gate, up, down) in every transformer block.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def add_lora(model, r=16, alpha=32):
    """Freeze the base model and attach trainable LoRA adapters.

    r is the rank (size) of each adapter; alpha scales its effect.
    """
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0,
                     task_type="CAUSAL_LM", target_modules=TARGET_MODULES)
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def finetune_memorize(model, tok, docs, epochs=3, lr=2e-4, batch_size=8,
                      max_len=128, seed=0, log_every=20):
    """Plain next-token training on the raw directory documents.

    This mimics how PII ends up in a model during pretraining: the model just
    reads text containing names and emails, with loss on every token. There is
    no question/answer format here -- the quiz question is never trained on.

    Returns a list of (step, loss) pairs for plotting the loss curve.
    """
    rng = random.Random(seed)
    device = next(model.parameters()).device

    # Only the LoRA parameters have requires_grad=True, so only they get updated.
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    # Mixed precision on GPU: bf16 if the GPU supports it, else fp16 (T4).
    # fp16 needs a GradScaler so small gradients don't round to zero.
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    # End each document with the end-of-text token so docs don't blur together.
    eos = tok.eos_token or ""
    texts = [d["text"] + eos for d in docs]

    # Learning rate decays linearly from lr to 0 over training.
    total_steps = epochs * math.ceil(len(texts) / batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1 - s / max(total_steps, 1))

    model.train()
    history, step = [], 0
    for ep in range(epochs):
        rng.shuffle(texts)  # new order every epoch
        for i in range(0, len(texts), batch_size):
            batch = tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                        truncation=True, max_length=max_len,
                        add_special_tokens=False).to(device)

            # labels = the input itself: the model learns to predict each next
            # token. Padding positions are set to -100 so they're ignored.
            labels = batch["input_ids"].clone()
            labels[batch["attention_mask"] == 0] = -100

            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                loss = model(**batch, labels=labels).loss

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)  # guard against spikes
            scaler.step(opt)
            scaler.update()
            sched.step()

            step += 1
            history.append((step, loss.item()))
            if step % log_every == 0:
                print(f"epoch {ep + 1} step {step}/{total_steps} loss {loss.item():.4f}")
    model.eval()
    return history
