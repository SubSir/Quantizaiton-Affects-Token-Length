import os
import json
from pathlib import Path
from datasets import load_dataset
from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    MultilingualExtractiveMatchMetric,
)
from lighteval.utils.language import Language
from lighteval.tasks.requests import Doc
from lighteval.models.model_output import ModelResponse


def get_gt_answers(task_name):
    if task_name == "aime90":
        ds = load_dataset("xiaoyuanliu/AIME90", split="train")
        gt_dict = {str(item["id"]): f"${item['answer']}$" for item in ds}
        return gt_dict
    elif task_name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        gt_dict = {str(i): f"${item['answer']}$" for i, item in enumerate(ds)}
        return gt_dict
    elif task_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        gt_dict = {
            str(i): f"${item['answer'].split('#### ')[-1]}$"
            for i, item in enumerate(ds)
        }
        return gt_dict
    else:
        raise ValueError(f"Unknown task: {task_name}")


english_metric = MultilingualExtractiveMatchMetric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(
        ExprExtractionConfig(),
        LatexExtractionConfig(boxed_match_priority=0),
    ),
    pred_extraction_target=(
        ExprExtractionConfig(),
        LatexExtractionConfig(boxed_match_priority=0),
    ),
    aggregation_function=max,
)

chinese_metric = MultilingualExtractiveMatchMetric(
    language=Language.CHINESE,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(
        ExprExtractionConfig(),
        LatexExtractionConfig(boxed_match_priority=0),
    ),
    pred_extraction_target=(
        ExprExtractionConfig(),
        LatexExtractionConfig(boxed_match_priority=0),
    ),
    aggregation_function=max,
)


def evaluate_output(model_output: str, gold_answer: str) -> bool:
    doc = Doc(
        query="dummy",
        choices=[gold_answer],
        gold_index=0,
        task_name="eval",
        id="0",
    )
    model_resp = ModelResponse(text=[model_output])

    result_en = english_metric.compute(doc=doc, model_response=model_resp)
    if result_en > 0:
        return True

    result_zh = chinese_metric.compute(doc=doc, model_response=model_resp)
    return result_zh > 0


def main():
    results_dir = Path("results")
    output_dir = Path("results_evaluated")
    output_dir.mkdir(exist_ok=True)

    for file_path in results_dir.glob("*.jsonl"):
        stem = file_path.stem
        if "_aime90" in stem:
            task = "aime90"
        elif "_math500" in stem:
            task = "math500"
        elif "_gsm8k" in stem:
            task = "gsm8k"
        else:
            print(f"Skipping unknown file: {file_path}")
            continue

        output_file = output_dir / f"{file_path.stem}.jsonl"

        if output_file.exists():
            print(f"Skipping {file_path} (already evaluated)")
            continue

        print(f"Processing {file_path} for task {task}...")
        gt_answers = get_gt_answers(task)

        with open(file_path, "r", encoding="utf-8") as fin, open(
            output_file, "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                data = json.loads(line.strip())
                sample_id = data.get("id", None)
                if sample_id is None:
                    raise ValueError(f"Missing 'id' in line: {line}")

                gold = gt_answers.get(str(sample_id))
                if gold is None:
                    print(
                        f"Warning: id {sample_id} not found in ground truth for {task}"
                    )
                    is_correct = False
                else:
                    model_output = data.get("output", "")
                    is_correct = evaluate_output(model_output, gold)

                data["is_correct"] = is_correct
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")

    print("✅ All files processed. Results saved in 'results_evaluated/'")


if __name__ == "__main__":
    main()
