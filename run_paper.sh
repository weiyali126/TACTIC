# run tactic
python -m tactic.cli gen-eval --config configs/tactic_paper/gen_eval-tactic-qwen2_5_7b.yaml
python -m tactic.cli gen-eval --config configs/tactic_paper/gen_eval-tactic-qwen2_5_14b.yaml
python -m tactic.cli gen-eval --config configs/tactic_paper/gen_eval-tactic-qwen2_5_32b.yaml
python -m tactic.cli gen-eval --config configs/tactic_paper/gen_eval-tactic-qwen2_5_72b.yaml
python -m tactic.cli gen-eval --config configs/tactic_paper/gen_eval-tactic-deepseek_v3.yaml

# run tactic-reasoning
python -m tactic.cli gen-eval --config configs/tactic_paper/gen_eval-tactic_reasoning-32b_r1.yaml
python -m tactic.cli gen-eval --config configs/tactic_paper/gen_eval-tactic_reasoning-32b_qwq32b.yaml
