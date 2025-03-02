# 针对多模态智能体工具交互能力的微调研究

# 初始设置

## 环境下载
在anaconda下创建用于执行的环境
```bash
conda create -n tongagent python=3.10
conda activate tongagent

pip install -r requirements.txt
```

如果想自己生成数据, 需要下载下面的环境.
```bash
pip install -r requirements_generation.txt
```

## 数据生成
生成数据时，运行下面的指令. 
```bash
bash data_generation.sh
```

## 准备数据集
收集的数据集打包为zip文件，直接放入data文件夹下

## 下载视觉工具的模型参数
你只需要**手动下载 SAM 2**，其他模型`transformers`会自动下载。
请将 `model_checkpoints` 文件夹放置在项目的根目录，使目录结构如下：
```
main.py
model_checkpoints/sam2_checkpoints
model_checkpoints/sam2_configs
```
你可以通过官方仓库提供的脚本下载模型权重和配置文件：
* [sam2_checkpoints](https://github.com/facebookresearch/sam2/blob/main/checkpoints/download_ckpts.sh)
* [sam2_configs](https://github.com/facebookresearch/sam2?tab=readme-ov-file#sam-21-checkpoints)

### 设置 Google 自定义搜索
该项目使用 **Google Customized Search** 进行网页搜索。你需要在 `configs/agent_config.yaml` 文件中设置 `cx` 和 `key`，它们位于 `search_engine` 部分：
```yaml
search_engine:
  -
    cx: # enter your cx here
    key: # enter your key here
```
获取 cx 和 key 的方法请参考[here](https://console.cloud.google.com/apis/api/customsearch.googleapis.com). 免费用户每天有 100 次查询的限制，付费用户每天最多可查询 10,000 次。

# Execute with closed-source api
## Setup
First, you need to set the api key and endpoint in `configs/agent_config.yaml`. The config file looks like this:
```yaml
tonggpt:
  model_name:  gpt-4o-2024-08-06
  region: eastus
  api_key: # enter your api key here
  open_ai_client_type: openai # or azure
  endpoint: # only for azure, you need to specify the endpoint you are using

agent_controller:
  engine_type: tonggpt # use minicpm, qwen if you want to use other models
```
We use GPT on Azure and provide a simple alternative for you to use original OpenAI client.

## Download benchmark dataset
You can download the GTA dataset from [GTA Link](https://github.com/open-compass/GTA/releases/download/v0.1.0/gta_dataset.zip), and revise your dataset path `data/gta_dataset/dataset.json` in `examples/gta/main.py` if you put it in some other path.

## Run

Run in command line manner with arbitrary prompt.
```bash
python main.py --prompt 'Can you edit the image to turn him into cyborg? Image path: tests/data/draw.jpg.'
```

See results runing on GTA set
```bash
python examples/gta/main.py
```

# Experiments
## Qwen-VL
Refer to official repo [Qwen-VL](https://github.com/QwenLM/Qwen2-VL) for environment setup.


After setup the environment, you can run the script convert dataset from MiniCPM-V to Qwen-VL format:
```bash
cd experiments/Qwen-VL

python scripts/convert_dataset_v2.py
```
Then you can run the script to train the model:
```bash
bash slurm_jobs/train_gaia.sh
bash slurm_jobs/train_gta.sh
```

## Evaluation
To evaluate the model, first modify the `configs/agent_config.yaml` to set the model path. Then run the script:
```bash
export RUN_MODE=eval

# for GAIA dataset
python examples/gaia/main.py --engine minicpm --lora-path experiments/CPM-FT/output/cpm_v2_6_7904295_2024_12_10_23_05/ --data-name 2023_level1 --split validation

python examples/gaia/main.py --engine minicpm --lora-path experiments/CPM-FT/output/cpm_v2_6_7904295_2024_12_10_23_05/ --data-name 2023_level2 --split validation

python examples/gaia/main.py --engine minicpm --lora-path experiments/CPM-FT/output/cpm_v2_6_7904295_2024_12_10_23_05/ --data-name 2023_level3 --split validation
# for GTA dataset
python examples/gta/main.py --engine minicpm --lora-path experiments/CPM-FT/output/cpm_v2_6_7904295_2024_12_10_23_05/
```
`cpm_v2_6_7904295_2024_12_10_23_05` is the model path. The training script automatically saves the model to that path. We use SLURM in our cluster such that the path consists of the job id and the time of the job. You should check the training script for the exact path.

Both benchmarks will output the results in `.cache` folder. You should use `eval.py` to get the metric we reported in the paper.

```bash
python examples/gaia/eval.py --data-path .cache/qa_cache/validation/minicpm/experiments/CPM-FT/output/cpm_v2_6_7904295_2024_12_10_23_05/2023_level1.db

python examples/gta/eval.py --folder .cache/gta/cpm_v2_6_7904295_2024_12_10_23_05/
```

# Acknowledgement
Thanks for their brilliant contributions to the community! Here are the codebases we built upon.

Our agent is based on the wonderful Huggingface Agent framework.
* https://huggingface.co/docs/transformers/v4.47.1/en/main_classes/agent#transformers.ReactCodeAgent

Our agent design is inspired by the following works:
* https://github.com/aymeric-roucher/GAIA
* https://github.com/Ag2S1/Sibyl-System

Model training and inference code:
* https://github.com/OpenBMB/MiniCPM-V
* https://github.com/QwenLM/Qwen2-VL
      eprint={2412.15606},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2412.15606}, 
}
```
