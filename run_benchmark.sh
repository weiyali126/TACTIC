# run zeroshot, fewshot baselines
python -m tactic.cli gen-eval --config configs/reference_benchmark/gen_eval-zeroshot-fewshot-gpt4_1.yaml
python -m tactic.cli gen-eval --config configs/reference_benchmark/gen_eval-zeroshot-fewshot-qwen.yaml
python -m tactic.cli gen-eval --config configs/reference_benchmark/gen_eval-zeroshot-fewshot-qwq_32b.yaml
python -m tactic.cli gen-eval --config configs/reference_benchmark/gen_eval-zeroshot-fewshot-deepseek_v3.yaml
python -m tactic.cli gen-eval --config configs/reference_benchmark/gen_eval-zeroshot-fewshot-deepseek_r1.yaml
