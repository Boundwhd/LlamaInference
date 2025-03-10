from transformers import AutoModelForCausalLM, AutoTokenizer

model = "/home/whd/llma3_pro/Llama-3.2-3B"

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model,
    device_map="auto",
    trust_remote_code=True,
).eval()
print(model)
inputs = tokenizer("hello", return_tensors="pt")
inputs = inputs.to(model.device)
pred = model.generate(
    **inputs, 
    max_length=100, 
    do_sample=True,         # 开启采样，避免重复
    temperature=0.8,        # 适当控制随机性
    top_p=0.9               # 只考虑累积概率前90%的token
)
test = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
print(test)

