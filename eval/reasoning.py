import json
from datasets import load_dataset
from vllm import LLM, SamplingParams
import torch
from tqdm import tqdm
import os
import argparse
import csv

max_new_tokens = 32768
# max_new_tokens = 1024
temperature = 0.0

def eval(model_name, task_name):
    num_gpus = torch.cuda.device_count()
    llm = LLM(model=model_name, tensor_parallel_size=num_gpus, gpu_memory_utilization=0.9)
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
    )

    if task_name == "aime90":
        ds = load_dataset("xiaoyuanliu/AIME90", split="train")
    elif task_name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    elif task_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
    else:
        raise NotImplementedError

    tasks = []
    for i, item in enumerate(ds):
        # if i > 50:
        #     break
        if "question" in item:  
            prompt = item["question"]
        elif "problem" in item:  
            prompt = item["problem"]
        else:
            continue
        tasks.append({"id": str(i), "prompt": prompt})

    print(f"Loaded {len(tasks)} tasks")

    results = []
    total_output_len = 0
    max_len_count = 0
    prompts = [task["prompt"] for task in tasks]
    outputs = llm.generate(prompts, sampling_params)
    total_output_len = 0
    max_len_count = 0
    results = []

    for task, output in tqdm(zip(tasks, outputs), total=len(tasks), desc="Processing outputs"):
        output_text = output.outputs[0].text
        prompt_len = len(output.prompt_token_ids)
        completion_len = len(output.outputs[0].token_ids)
        total_len = prompt_len + completion_len

        result = {
            "id": task["id"],
            "prompt": task["prompt"],
            "output": output_text,
            "prompt_tokens": prompt_len,
            "completion_tokens": completion_len,
            "total_tokens": total_len,
        }
        total_output_len += completion_len
        if completion_len == max_new_tokens:
            max_len_count += 1
        results.append(result)

    os.makedirs("results", exist_ok=True)
    out_file = f"results/{model_name.split('/')[-1]}_{task_name}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved all results to {out_file}")

    csv_file = "evaluation_summary.csv"
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["model_name", "task_name", "total_completion_tokens", "max_len_count"])
        writer.writerow([model_name.split('/')[-1], task_name, total_output_len, max_len_count])

    print(f"Appended summary to {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="path of the model")
    parser.add_argument("--task", type=str, help="tasks to evaluate")
    args = parser.parse_args()

    eval(args.model, args.task)

