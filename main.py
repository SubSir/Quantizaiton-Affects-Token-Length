from eval import reasoning, wikitext

models = ["/localssd/zhzhang/DeepSeek-R1-Distill-Qwen-7B"]
bit_range = [4]
group_range = [64, 128]
tasks = ["aime90", "gsm8k", "math500"]
import multiprocessing as mp

def run_eval(model_path, tasks):
    wikitext.eval(model_path)
    reasoning.eval(model_path, tasks)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    for base_model in models:
        for bit in bit_range:
            for group in group_range:
                model_path = f"{base_model}-w{bit}g{group}"
                print(f"=== Running {model_path} ===")

                p = mp.Process(target=run_eval, args=(model_path, tasks))
                p.start()
                p.join()  
                
                if p.exitcode != 0:
                    print(f"Process failed for {model_path}, exit code {p.exitcode}")
