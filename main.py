from eval import reasoning, wikitext
models = ["/localssd/zhzhang/DeepSeek-R1-Distill-Qwen-1.5B"]
bit_range = [3, 4]
group_range = [64, 128]
tasks = ["aime90", "gsm8k", "math500"]

for base_model in models:
  for bit in bit_range:
    for group in group_range:
      model_path = f"{base_model}-w{bit}g{group}"
      wikitext.eval(model_path)
      for task in tasks:
        reasoning.eval(model_path, task)