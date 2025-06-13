<div align="center">
    
## TACTIC: Translation Agents with Cognitive-Theoretic Interactive Collaboration

<!-- **Authors:** -->

**Weiya Li**<sup>1</sup>, **Junjie Chen**<sup>2</sup> <sup>†</sup>, **Bei Li**<sup>3</sup>, **Boyang Liu**<sup>4</sup>, **Zichen Wen**<sup>2</sup>, **Nuanqiao Shan**<sup>5</sup>, **Xiaoqian Liu**<sup>2,6</sup>, **Anping Liu**<sup>1</sup>, **Huajie Liu**<sup>1</sup>, **Hu Song**<sup>1</sup>, **Linfeng Zhang**<sup>2</sup> <sup>★</sup>

<!-- **Affiliations:** -->

<sup>1</sup> Big Data & AI Lab, ICBC, <sup>2</sup> Shanghai Jiao Tong University, <sup>3</sup> Meituan Inc., <sup>4</sup> Tongji University, <sup>5</sup> Fudan University, <sup>6</sup> Northeastern University

📝 [**Paper**](https://arxiv.org/abs/2506.08403) 

</div>

<sup>★</sup> *Corresponding author.*

## Overview
**TACTIC** is a cognitively informed multi-agent translation framework that emulates human translation behavior. It models six distinct cognitive roles—**drafting**, **refinement**, **evaluation**, **scoring**, **context reasoning**, and **external knowledge gathering**—to enhance translation quality using large language models (LLMs).

<div align="center">
  <img src="/docs/main-workflow.png" alt="main workflow" width="800">
</div>

## Release Notes❗
- Paper released on June 10, 2025; more code will be open-sourced upon acceptance.

## Contents

- [Key Features](#key-features)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Deploy online services](#deploy-online-services)
- [Directory Structure](#directory-structure)


## Key Features <a name="key-features"></a>
- **Cognitive-Theoretic Architecture**: Grounded in Cognitive Translation Studies (CTS), aligning with key concepts like cognitive strategies, processing, and contextual cognition.
- **Modular Agent System**: Includes `DraftAgent`, `RefinementAgent`, `EvaluationAgent`, `ScoreAgent`, `ContextAgent`, and `ResearchAgent`.
- **Iterative Translation Workflow**: Combines fast initial translation with context-enhanced refinement for complex cases.
- **SOTA Performance**: Outperforms GPT-4.1 and DeepSeek-R1 on FLORES-200 and WMT24 benchmarks.
- **Multi-Language Support**: Any language pairs supported by the base models are compatible. We use the following languages as examples: English ↔️ Chinese, German, Japanese, Russian, and Ukrainian.

| Agent            | Cognitive Concepts           | Description                                                      |
|------------------|------------------------------|------------------------------------------------------------------|
| `DraftAgent`     | Cognitive Strategies         | Produces literal, sense-for-sense, and free drafts               |
| `RefinementAgent`| Cognitive Processing         | Synthesizes drafts into a polished output                        |
| `EvaluationAgent`| Cognitive Processing         | Evaluates translations on faithfulness, expressiveness, elegance |
| `ScoreAgent`     | Cognitive Processing         | Scores and decides if threshold is met                           |
| `ContextAgent`   | Contextual Cognition         | Supplies situational and discourse context                       |
| `ResearchAgent`  | Contextual Cognition         | Extracts domain-specific terms and collocations                  |


## Results <a name="results"></a>

### Table 1: Evaluation Results on the WMT24 Test Set

| Task      | Models   | XCOMET<br>(en→xx)   | KIWI-23<br>(en→xx)    | XCOMET<br>(xx→en)    | KIWI-23<br>(xx→en)    |
|:------------|:-------------:|:---------------------:|:-----------------------:|:----------------------:|:-----------------------:|
| Zero-shot | Qwen2.5-14B    |                  73.97 | 68.57                   | 82.59                  | 77.83                   |
| Zero-shot | Qwen2.5-32B    |                  76.21 | 70.87                   | 84.37                  | 79.4                    |
| Zero-shot | GPT-4.1        |                  86.83 | 81.03                   | 87.73                  | 81.33                   |
| Zero-shot | DeepSeek-V3    |                  84.71 | 79.61                   | 87.2                   | 80.99                   |
| Zero-shot | DeepSeek-R1    |                  86.1  | 80.45                   | 87.98                  | 75.01                   |
| Few-shot  | Qwen2.5-14B    |                  75.03 | 69.62                   | 82.66                  | 77.93                   |
| Few-shot  | Qwen2.5-32B    |                  76.61 | 71.22                   | 84.36                  | 79.48                   |
| Few-shot  | GPT-4.1        |                  86.92 | 81.07                   | 87.9                   | 81.39                   |
| Few-shot  | DeepSeek-V3    |                  85.58 | 80.09                   | 87.05                  | 81.04                   |
| TACTIC    | Qwen2.5-14B    |                  81.12 | 76.25                   | 86.1                   | 80.74                   |
| TACTIC    | Qwen2.5-32B    |                  82.97 | 78.06                   | 87.13                  | 81.56                   |
| TACTIC    | DeepSeek-V3    |              **86.95** | **81.26**               | **89.07**              | **82.46**               |


## Installation <a name="installation"></a>

We recommend using the vllm image as the base environment to support local deployment. Please pull the latest version of the vLLM image:
```bash
docker pull vllm/vllm-openai:latest
```
Install tactic environment dependencies.
```bash
git clone https://github.com/yourname/TACTIC
cd TACTIC
pip install -r requirements.txt
```

## Usage <a name="usage"></a>

Please first configure the API KEY and BASE URL in `config.py`.  
**Run Examples.**
```python
## generate
python -m tactic.cli generate --config configs/generate.yaml

## evaluate
python -m tactic.cli evaluate --config configs/evaluate.yaml

## generate-evaluate
python -m tactic.cli gen-eval --config configs/gen_eval.yaml
```

**Replicating our work.** 

```sh
# Reproduce baselines
bash run_benchmark.sh

# Reproduce tactic paper
bash run_paper.sh
```
>Note: Due to the inherent randomness in LLM outputs, the final results may exhibit slight fluctuations.


## Deploy online services <a name="deploy-online-services"></a>
1. Deploy the back-end TACTIC service.
```sh
python -m tactic.app.server
```
2. Deploy the front-end web service.
```sh
python -m tactic.app.frontend
```
3. Visit http://localhost:9002/

<div align="center">
  <img src="/docs/web-demo.gif" alt="web-demo" width="800">
</div>


## Directory Structure <a name="directory-structure"></a>

```
tactic/
├── agents/                  # Translation agents implementing drafting, refinement, evaluation, etc.
│   ├── common_agent.py      # zeroshot, fewshot agents
│   ├── draft_agent.py       # Generates initial translation drafts
│   ├── refinement_agent.py  # Refines drafts into polished translations
│   ├── evaluation_agent.py  # Evaluates translations on faithfulness, expressiveness, elegance
│   ├── score_agent.py       # Score translations on faithfulness, expressiveness, elegance
│   ├── research_agent.py    # Retrieves domain-specific knowledge and keywords
│   └── context_agent.py     # Supplies contextual information to guide translation
│
├── metrics/                 # List of evaluation indicators
│   ├── xcomet 
│   ├──comet_kiwi_23        
│   ...  
│
├── models/                  # Configuration files and model parameters
│   ├── model_init.py        # Initialize the model
│   └── model_loader.py      # Configuration for vLLM backend
│
├── prompts/                 # Prompts management
│   ├── common.py            # Zeroshot, fewshot prompts
│   └── multi_agent.py       # Tactic agents prompts
|
├── tasks/                   # Agents calling functions
│   ├── generate.py          # The main function of agents
│   └── evaluate.py          # Standby evaluation function
│
├── app/                      # Entry point for online service
│   ├── server.py         # The main entrance at the back-end
│   ├── frontend.py           # Front-end web service
│   └── tactic.py             # tactic agents

├── cli.py                   # Command-line interface to interact with the system
├── config.py                # Configure API key, directories, and threshold constants
└── utils.py                 # Common functions
```

## Notes❗

[CAMEL](https://github.com/camel-ai/camel/tree/master) requires adding the `extra_body` field in `camel.configs.vllm_config` to support structured output. The version of CAMEL referenced in this project has already incorporated this modification.

```python
class VLLMConfig(BaseConfig):
    extra_body: dict  # add extra_body support
```

## Acknowledgments

Our project is built upon [CAMEL](https://github.com/camel-ai/camel/tree/master) and [Tower-Eval](https://github.com/deep-spin/tower-eval), and we sincerely thank them for their contributions to the open-source community.

## Contact

Questions or feedback? Open an issue or email [weiyali126@outlook.com](mailto:weiyali126@outlook.com).

All contributions welcome! All content in this repository is licensed under the MIT license.

## Citation

If you use this work, please cite our paper:

```bibtex
@misc{li2025tactictranslationagentscognitivetheoretic,
      title={TACTIC: Translation Agents with Cognitive-Theoretic Interactive Collaboration}, 
      author={Weiya Li and Junjie Chen and Bei Li and Boyang Liu and Zichen Wen and Nuanqiao Shan and Xiaoqian Liu and Anping Liu and Huajie Liu and Youyan Wang and Wujiuge Yin and Hu Song and Bing Huang and Zhiyuan Xia and Jialiang Chen and Linfeng Zhang},
      year={2025},
      eprint={2506.08403},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.08403}, 
}
```
