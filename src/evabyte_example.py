from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# BLAH

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("evabyte/EvaByte", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("evabyte/EvaByte", torch_dtype=torch.bfloat16, trust_remote_code=True).eval().to("cuda")
model.eval()
prompt = "That’s one small ste"

input_ids = torch.tensor([[1] + [b + 64 for b in prompt.encode("utf-8")]]).to("cuda")
seq_len = input_ids.shape[1]
position_ids = torch.arange(seq_len, dtype=torch.long, device="cuda").unsqueeze(0)

with torch.no_grad():
    outputs = model(input_ids=input_ids, position_ids=position_ids)
    logits = outputs[0]  # shape: [1, seq_len, vocab_size]

# Get logits for the next byte/token
next_logits = logits[0, -1, :]  # [vocab_size]

# Get top-3 byte predictions
topk = torch.topk(next_logits, k=3)
top_ids = topk.indices.tolist()
top_logits = topk.values.tolist()

# Convert token IDs back to bytes/chars
# EvaByte decodes by subtracting 64 from the token ID (inverse of encoding)
top_bytes = [bytes([token_id - 64]) for token_id in top_ids]

# Print results
for i in range(3):
    b = top_bytes[i]
    try:
        char = b.decode("utf-8")
    except UnicodeDecodeError:
        char = "�"  # Replacement char
    print(f"Rank {i+1}: byte={repr(b)} char='{char}' (id={top_ids[i]}, logit={top_logits[i]:.2f})")



# Tokenize input
# Option 1: standard HF tokenizer interface
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

# Option 2: Direct UTF-8 byte encoding with offset
# Note: Each byte is offset by 64 with <bos> prepended.
input_ids = torch.tensor([[1] + [b + 64 for b in prompt.encode("utf-8")]]).to("cuda")

with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs[0]  # First item in the tuple is logits

# Get logits for the next byte/token
next_logits = logits[0, -1, :]  # [vocab_size]

# Get top-3 byte predictions
topk = torch.topk(next_logits, k=3)
top_ids = topk.indices.tolist()
top_logits = topk.values.tolist()

# Convert token IDs back to bytes/chars
top_bytes = tokenizer.convert_ids_to_tokens(top_ids)

# Print results
for i in range(3):
    token = top_bytes[i]
    token_display = repr(token.encode("latin1")) if isinstance(token, str) else str(token)
    print(f"Rank {i+1}: byte={token_display} (id={top_ids[i]}, logit={top_logits[i]:.2f})")

# byte-by-byte generation (default)
generation_output = model.generate(
    input_ids=input_ids, 
    max_new_tokens=10,
    num_beams=3,
    num_return_sequences=3,
    early_stopping=True
)

    # num_return_sequences=3,
    # num_beams=3
    #do_sample=True,               # Enable sampling
    # top_k=50,                     # Optional: top-k sampling
    # top_p=0.95,                   # Optional: nucleus sampling
    # num_return_sequences=3,       # ✅ Get top 3 different outputs

# alternatively, use faster multibyte generation
generation_output = model.multi_byte_generate(
    input_ids=input_ids, 
    max_new_tokens=1,
)

# Decode and print the output
for output in generation_output:
    prediction = tokenizer.decode(
        output[input_ids.shape[1]:], 
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    print(prediction)

    
"""
response = tokenizer.decode(
    generation_output[0][input_ids.shape[1]:], 
    skip_special_tokens=False,
    clean_up_tokenization_spaces=False,
)
"""
# print(response)
# Sample output:
# over the lazy dog.\n\nThe quick
