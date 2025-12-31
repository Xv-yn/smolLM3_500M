import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    "./runs/poc_from_text/ckpt-step-500",
    trust_remote_code=True,
    fix_mistral_regex=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "./runs/poc_from_text/ckpt-step-500",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

inputs = tok("Once upon a time", return_tensors="pt").to(model.device)
out = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.8,
)
print(tok.decode(out[0], skip_special_tokens=True))
