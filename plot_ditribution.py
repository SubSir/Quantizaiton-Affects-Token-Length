import json
import os
import glob
import matplotlib.pyplot as plt

def plot_completion_tokens_distribution(jsonl_file: str, save_dir: str = "png") -> None:
    """
    读取单个 JSONL 文件并绘制 completion_tokens 分布图，保存到指定目录。
    """
    os.makedirs(save_dir, exist_ok=True)

    # 生成输出文件名：原文件名把 .jsonl 换成 .png
    base_name = os.path.basename(jsonl_file).replace(".jsonl", ".png")
    save_path = os.path.join(save_dir, base_name)

    completion_tokens_list = []

    # 读取 JSONL
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                completion_tokens = data.get("completion_tokens", 0)
                completion_tokens_list.append(completion_tokens)
            except json.JSONDecodeError:
                print(f"[WARN] {jsonl_file}:{line_no} 行解析失败，已跳过")
                continue

    if not completion_tokens_list:
        print(f"[WARN] {jsonl_file} 未提取到任何 completion_tokens，跳过绘图")
        return

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.hist(completion_tokens_list, bins=50, color="skyblue", edgecolor="black")
    plt.title(f"Distribution of Completion Tokens\n({os.path.basename(jsonl_file)})")
    plt.xlabel("Completion Tokens")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] 图像已保存到 {save_path}")

def batch_plot_all_jsonl_under_results(results_dir: str = "results") -> None:
    """
    批量处理 results/ 目录下所有 .jsonl 文件
    """
    pattern = os.path.join(results_dir, "*.jsonl")
    files = glob.glob(pattern)

    if not files:
        print(f"[INFO] 在 {results_dir}/ 下未找到任何 .jsonl 文件")
        return

    for jsonl in files:
        print(f"[INFO] 正在处理 {jsonl} ...")
        plot_completion_tokens_distribution(jsonl)

if __name__ == "__main__":
    batch_plot_all_jsonl_under_results("results")