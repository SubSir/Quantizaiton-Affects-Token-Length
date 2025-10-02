# pseudo quantize script for awq
python quantize/awq/run_awq.py \
  --model /localssd/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --w_groupsize 64 \
  --w_asym \
  --save_qmodel_path /localssd/zhzhang/DeepSeek-R1-Distill-Qwen-1.5B-w3g64 \
  --w_bits 3