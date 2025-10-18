<p align="center" style="margin:0"><img src="./assets/logo2.png" alt="agent" style="width: 45%" /></p>\
<h2 align="center">From charts to code: a hierarchical benchmark for multimodal models</h2>

<h5 align="center"> 
Welcome to Chart2Code! If you find this repo useful, please give a star ⭐ for encouragement.
</h5>

<h5 align="center"> 
<!-- 
English | [中文](README_zh.md) -->

[![arXiv](https://img.shields.io/badge/ArXiv-2502.08047-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2502.08047)  [![Project Page](https://img.shields.io/badge/Project_Page-WorldGUI-white?color=7289DA)](https://showlab.github.io/WorldGUI) [![Project Page](https://img.shields.io/badge/Github-AwesomeGUI-blue)](https://github.com/showlab/Awesome-GUI-Agent)

</h5>



## 🌟 Overview

**Chart2Code-Benchmark** is a new benchmark designed to evaluate chart generation capabilities of LMMs under progressively challenging conditions. 
<p align="center" style="margin:0"><img src="./assets/figure1.png" alt="agent" style="width: 85%" /></p>

**Chart2Code covers three progressively challenging levels:** reproduction, editing, and long-table to chart generation. <br>Level1(<b>Chart Reproduction</b>) reproduces charts from a reference figure and user query; <br>Level2(<b>Chart Editing</b>) involves complex modifications such as changing chart types or adding elements;<br>Level3(<b>Long-Table to Chart Generation</b>) requires models to transform long, information-dense tables into faithful charts following user instructions.

More details about Chart2Code in [project page](https://showlab.github.io/WorldGUI).🌐



## 🚀 Quick Start
Here we provide a quick start guide to evaluate LMMs on Chart2Code.

### Setup Environment
```shell
git clone https://github.com/showlab/Chart2Code.git
conda env create -f environment.yaml
conda activate chart2code
cd Chart2Code
```

Setup API key and API base URL in `.env` for different LMMs. Claude、 Gemini and GPT are accessed through API proxy providers，while Seed is accessed through ARK API.
```shell
OPENAI_API_KEY=${your_api_proxy_provider_key}
ARK_API_KEY=${your_ark_api_key}
OPENAI_API_URL=${your_api_proxy_provider_url}
ARK_BASE_URL=${your_ark_api_base_url}
```
### Download Data
Download the Chart2Code data from [Huggingface](https://huggingface.co/datasets/hhenryz/Chart2Code-Benchmark) and unzip it under the root directory.
```shell
cd data
wget https://huggingface.co/datasets/hhenryz/Chart2Code-Benchmark/resolve/main/Chart2Code-Benchmark.zip
unzip Chart2Code-Benchmark.zip
cd ..


```
The file structure should be like this:
```Chart2Code
├── data
│   ├── level1_direct
│   │   ├── 3d_1.png
│   │   ├── 3d_1.py
│   │   └── ...
│   ├── level1_figure
│   │   ├── fig1_density_2
│   │   ├── ...
│   └── level1_customize
│       ├── table_1_instruction_2.png
│       ├── table_1_instruction_2.py
│       ├── table_1_instruction_2_request.txt
│       └── table_1_instruction_2_data.txt
│       └── ...
│   ├── level2
│   │   ├── bar_1_v1.png
│   │   ├── bar_1_v1.py
│   │   ├── bar_1_v1_data.txt
│   │   └── ...
│   └── level3
│       ├── table_1.xlsx
│       ├── table1_1.png
│       ├── table1_1_generate.py
│       ├── table1_1.txt
│       ├── table1_1_generate.png
│       └── ...
│   ├── level1_direct.json
│   ├── level1_figure.json
│   ├── level1_customize.json
│   ├── level2.json
│   └── level3.json
│—— Evaluation
└── ...
```

### Inference Setup

Inference for each benchmark level is handled by a dedicated shell script located in the `scripts/` directory.
You must specify a model for each run. You can do this in two ways:
-   Pass it as an argument (Recommended): Provide the `MODEL_IDENTIFIER` directly when executing the script.
-   Edit the script: Set the `MODEL_IDENTIFIER` variable inside the corresponding .sh file.

You can modify the `LOAD_SOURCE` parameter in the shell script to select how the model is loaded:
-   `local`: By default, the model will be loaded from the `Inference/models` directory.
-   `hub`: The model weights will be loaded directly from the Hugging Face Hub online.
   
You can also adjust other parameters like `GPU_VISIBLE_DEVICES`  in the script to fit your hardware setup.
```shell
cd scripts/inference
# For level1_customize
bash inference_customize.sh qwen3_customize_30B
# For level1_direct
bash inference_direct.sh qwen2.5_direct_72B
# For level1_figure
bash inference_figure.sh InternVL_3.5_figure_38B
# For level2
bash inference_level2.sh deepseek_level2
# For level3
bash inference_level3.sh gpt_5_level3
```
<details>
<summary>Available Models</summary>
We now support the following models:
<table>
  <tr>
    <th rowspan="2">Model Name</th>
    <th colspan="5">MODEL_IDENTIFIER</th>

  </tr>
  <tr>
    <th>level1_customize</th>
    <th>level1_direct</th>
    <th>level1_figure</th>
    <th>level2</th>
    <th>level3</th>
  </tr>
  <tr>
    <td>Qwen3-VL-30B</td>
    <td> qwen3_customize_30B</td>
    <td>qwen3_direct_30B</td>
    <td>qwen3_figure_30B</td>
    <td>qwen3_level2_30B</td>
      <td>qwen3_level3_30B</td>
  </tr>
  <tr>
    <td>Qwen2.5-VL-72B</td>
    <td>qwen2.5_customize_72B</td>
    <td>qwen2.5_direct_72B</td>
    <td>qwen2.5_figure_72B</td>
    <td>qwen2.5_level2_72B</td>
    <td>qwen2.5_level3_72B</td>
  </tr>
  <tr>
    <td>InternVL-3.5-38B</td>
      <td>InternVL_customize_38B</td>
      <td>InternVL_direct_38B</td>
      <td>InternVL_figure_38B</td>
      <td>InternVL_level2_38B</td>
      <td>InternVL_level3_38B</td>
  </tr>
    <tr>
    <td>InternVL-3.5-8B</td>
      <td>InternVL_customize_8B</td>
      <td>InternVL_direct_8B</td>
      <td>InternVL_figure_8B</td>
      <td>InternVL_level2_8B</td>
      <td>InternVL_level3_8B</td>
  </tr>
</table>
</details>

### Evaluate Setup

For the results obtained from inference, the first step is to check the execution rate. The code that runs successfully and its corresponding generated images will undergo the following evaluations: `base_evaluation`, `LLM_evaluation`, and `LMM_evaluation`.
```shell
cd scripts/evaluate
# step1: check execution rate
bash execute_evaluate.sh
# step2: run base evaluation
bash base_evaluator.sh
# step3: run LLM evaluation to evaluate the code
bash LLM_evaluator.sh
# step4: run LMM evaluation to evaluate the image
bash LMM_evaluator.sh
```
## 📢 Update

* [2025.02.13] We release the Chart2Code in [arxiv](https://arxiv.org/abs/2502.08047).


## ❤ Acknowledgement
- Special thanks to [Difei Gao](https://scholar.google.com/citations?user=No9OsocAAAAJ&hl=en) for his hard work on devleoping the codebase.

- We express our great thanks to Kaiming Yang, Mingyi Yan, Wendi Yu for their hard work for data ananotation and baseline testing.

- [OOTB (Computer Use)](https://github.com/showlab/computer_use_ootb?tab=readme-ov-file): Computer Use OOTB is an out-of-the-box (OOTB) solution for Desktop GUI Agent, including API-based (Claude 3.5 Computer Use) and locally-running models (ShowUI, UI-TARS).

- [ShowUI](https://github.com/showlab/ShowUI): Open-source, End-to-end, Lightweight, Vision-Language-Action model for GUI Agent & Computer Use.

- [AssistGUI](https://arxiv.org/pdf/2312.13108): AssistGUI is the first work that focuses on desktop productivity software usage with over 100 realistic GUI tasks.

- [VideoGUI](https://github.com/showlab/videogui): A Benchmark for GUI Automation from Instructional Videos. Can a GUI agent behave like a human when giving an image-style effect and a user query?


- [SWE-bench Multimodal](https://www.swebench.com/multimodal.html): SWE-bench Multimodal is a dataset for evaluating AI systems on visual software engineering tasks.

## 🎓 BibTeX

If you find WorldGUI useful, please cite using this BibTeX:

```bibtex
@misc{zhao2025worldguiinteractivebenchmarkdesktop,
      title={WorldGUI: An Interactive Benchmark for Desktop GUI Automation from Any Starting Point}, 
      author={Henry Hengyuan Zhao and Kaiming Yang and Wendi Yu and Difei Gao and Mike Zheng Shou},
      year={2025},
      eprint={2502.08047},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2502.08047}, 
}
```


## 🔔 Contact
If you have any questions or suggestions, please don't hesitate to let us know. You can directly email [Henry Hengyuan Zhao](https://zhaohengyuan1.github.io/) at NUS using the email address hubylidayuan@gmail.com, or post an issue on this repository. We welcome contributions. Feel free to submit pull requests if you have suggestions for improvement.

