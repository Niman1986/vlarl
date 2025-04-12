# VLA-RL: Toward Masterful and General Robotic Manipulation with Scalable Reinforcement Learning

<div align="center">

[![blog](https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=notion&logoColor=white)](https://arxiv.org/abs/2406.09246)

</div>

## 🌟 Highlights

- 🎯 **Masterful & General Manipulation**: We present VLA-RL, an open-source solution advancing vision-language-action (VLA) models with reinforcement learning beyond imitation learning.

- ⚡️ **Cutting-edge Architecture**: Built with ray+vllm+lora+fsdp, our codebase delivers both scalability and flexibility.

- 📝 **Clean Implementation**: Following [cleanrl](https://github.com/vwxyzjn/cleanrl)'s philosophy, we provide a single-file implementation for easy reading and modification.

- 🚧 **Active Development**: Work in Progress, let's build it together.

## 📝 TODO
- [ ] Support SERL-style Real-world RL
- [ ] Support More Environments (e.g., Roboverse)
- [ ] Support More VLAs

## 🛠️ Installation

See [INSTALL.md](docs/INSTALL.md) for installation instructions. 

See [ERROR_CATCH.md](docs/ERROR_CATCH.md) for error catching.

## 🚀 Quick Start

Before launching distributed training, please edit the script with the appropriate dataset and model paths first.

### 📈 Training

```bash
# bash train_rl_vllm_ray_fsdp.sh <gpus> <task_ids>
# e.g., 
bash train_rl_vllm_ray_fsdp.sh 6,7 0,1,2,3,4,5,6,7,8,9
```

### 🧪 Evaluation

```bash
# parallel evaluation with vectorized environment
bash eval_vec.sh
```

## 🏷️ License
This repository is released under the Apache-2.0 license.

## 🙏 Acknowledgement

Our code is built upon [open-instruct](https://github.com/allenai/open-instruct), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [verl](https://github.com/volcengine/verl), [splatter-image](https://github.com/szymanowiczs/splatter-image). We thank all these authors for their nicely open sourced code and their great contributions to the community.

## 🥰 Citation
If you find this repository helpful, please consider citing:

```
@misc{lu2025vlarl,
  title={VLA-RL: Toward Masterful and General Robotic Manipulation with Scalable Reinforcement Learning},
  author={Guanxing Lu, Yuheng Zhou, Haonan Jiang, Chubin Zhang, Zifeng Gao, Ziwei Wang and Yansong Tang},
  year={2025},
  howpublished={\url{https://congruous-farmhouse-8db.notion.site/VLA-RL-Toward-Masterful-and-General-Robotic-Manipulation-with-Scalable-Reinforcement-Learning-1953a2cd706280ecaad4e93a5bd2b8e3?pvs=4}},
  note={Notion Blog}
}
```
