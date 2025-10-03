import torch
import tqdm
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import csv
import torch.distributed as dist
import gc

def get_wikitext2(tokenizer):
    from datasets import load_dataset
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    return testenc

def eval(model_name):
    csv_file = "ppl_summary.csv"
    model_short_name = model_name.split('/')[-1]

    if os.path.isfile(csv_file):
        with open(csv_file, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("model_name") == model_short_name:
                    print(f"Already evaluated PPL for model: {model_name}. Skipping.")
                    return 
                
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model.seqlen = 2048
    testenc = get_wikitext2(tokenizer)
    
    testenc = testenc.input_ids.to(model.device)
    nsamples = testenc.numel() // model.seqlen
    model = model.eval()
    nlls = []
    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
            model.device
        )
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[
            :, (i * model.seqlen) : ((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())    

    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["model_name", "ppl"])
        writer.writerow([model_name.split('/')[-1], ppl.item()])

    print(f"Appended summary to {csv_file}")

    del model
    gc.collect()
    torch.cuda.empty_cache()