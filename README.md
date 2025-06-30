# BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation
[[paper]](https://arxiv.org/abs/2506.07530) [[model]](https://huggingface.co/collections/hongyuw/bitvla-68468fb1e3aae15dd8a4e36e) [[code]](https://github.com/ustcwhy/BitVLA)

- June 2025: [BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation](https://arxiv.org/abs/2506.07530)


## Open Source Plan

- ✅ Paper, Pre-trained VLM and evaluation code.
- ✅ Fine-tuned VLA code and models
- 🧭 Pre-training code and VLA.

## Contents

- [BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation](#bitvla-1-bit-vision-language-action-models-for-robotics-manipulation)
  - [Contents](#contents)
  - [Checkpoints](#checkpoints)
  - [Vision-Language](#vision-language)
    - [Evaluation on VQA](#evaluation-on-vqa)
  - [Vision-Language-Action](#vision-language-action)
    - [OFT Training](#oft-training)
      - [1. Preparing OFT](#1-preparing-oft)
      - [2. OFT fine-tuning](#2-oft-fine-tuning)
    - [Evaluation on LIBERO](#evaluation-on-libero)
  - [Acknowledgement](#acknowledgement)
  - [Citation](#citation)
  - [License](#license)
    - [Contact Information](#contact-information)
   
## Checkpoints

| Model     | Path |
| -------------- | ----- |
| BitVLA |   [hongyuw/bitvla-bitsiglipL-224px-bf16](https://huggingface.co/hongyuw/bitvla-bitsiglipL-224px-bf16)    |
| BitVLA finetuned on LIBERO-Spatial |   [hongyuw/ft-bitvla-bitsiglipL-224px-libero_spatial-bf16](https://huggingface.co/hongyuw/ft-bitvla-bitsiglipL-224px-libero_spatial-bf16)    |
| BitVLA finetuned on LIBERO-Object  |   [hongyuw/ft-bitvla-bitsiglipL-224px-libero_object-bf16](https://huggingface.co/hongyuw/ft-bitvla-bitsiglipL-224px-libero_object-bf16)    |
| BitVLA finetuned on LIBERO-Goal    |   [hongyuw/ft-bitvla-bitsiglipL-224px-libero_long-bf16](https://huggingface.co/hongyuw/ft-bitvla-bitsiglipL-224px-libero_long-bf16)    |
| BitVLA finetuned on LIBERO-Long    |   [hongyuw/ft-bitvla-bitsiglipL-224px-libero_long-bf16](https://huggingface.co/hongyuw/ft-bitvla-bitsiglipL-224px-libero_long-bf16)    |
| BitVLA w/ BF16 SigLIP |  [hongyuw/bitvla-siglipL-224px-bf16](https://huggingface.co/hongyuw/bitvla-siglipL-224px-bf16)     |

*Note that we provide the master weights of BitVLA and perform online quantization. For actual memory savings, you may quantize the weights offline to 1.58-bit precision. We recommend using the [bitnet.cpp](https://github.com/microsoft/bitnet) inference framework to accurately measure the reduction in inference cost.*

*Due to limited resources, we have not yet pre-trained BitVLA on a large-scale robotics dataset. We are actively working to secure additional compute resources to conduct this pre-training.*

## Vision-Language

### Evaluation on VQA

We use the [LMM-Eval](https://github.com/ustcwhy/BitVLA/tree/main/lmms-eval) toolkit to conduct evaluations on VQA tasks. We provide the [transformers repo](https://github.com/ustcwhy/BitVLA/tree/main/transformers) in which we modify the [modeling_llava.py](https://github.com/ustcwhy/BitVLA/blob/main/transformers/src/transformers/models/llava/modeling_llava.py) and [modeling_siglip.py](https://github.com/ustcwhy/BitVLA/blob/main/transformers/src/transformers/models/siglip/modeling_siglip.py) to support the W1.58-A8 quantization. 

The evaluation should use nvidia_24_07 docker. Install the packages:

```bash
docker run --name nvidia_24_07  --privileged --net=host --ipc=host --gpus=all -v /mnt:/mnt -v /tmp:/tmp -d nvcr.io/nvidia/pytorch:24.07-py3 sleep infinity # only use for multimodal evaluation
docker exec -it nvidia_24_07 bash
git clone https://github.com/ustcwhy/BitVLA.git
cd BitVLA/
bash vl_eval_setup.sh # only use for multimodal evaluation
```

First, download the BitVLA model from HuggingFace:

```bash
git clone https://huggingface.co/hongyuw/bitvla-bitsiglipL-224px-bf16 # BitVLA w/ W1.58-A8 SigLIP-L
git clone https://huggingface.co/hongyuw/bitvla-siglipL-224px-bf16 # BitVLA w/ BF16 SigLIP-L
```

Then run the following scripts to conduct evaluations:

```bash
cd lmms-eval/
bash eval-dense-hf.sh /YOUR_PATH_TO_EXP/bitvla-bitsiglipL-224px-bf16
bash eval-dense-hf.sh /YOUR_PATH_TO_EXP/bitvla-siglipL-224px-bf16
```

Note that we provide the master weights of BitVLA and perform online quantization. For actual memory savings, you may quantize the weights offline to 1.58-bit precision. We recommend using the [bitnet.cpp](https://github.com/microsoft/bitnet) inference framework to accurately measure the reduction in inference cost.

## Vision-Language-Action

### OFT Training 

#### 1. Preparing OFT
We fine-tune BitVLA using OFT training shown in [OpenVLA-OFT](https://github.com/moojink/openvla-oft/tree/main). First setup the environment as required by that project. You can refer to [SETUP.md](https://github.com/moojink/openvla-oft/blob/main/SETUP.md) and [LIBERO.md](https://github.com/moojink/openvla-oft/blob/main/LIBERO.md) for detailed instructions.

```
conda create -n bitvla python=3.10 -y
conda activate bitvla
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124

# or use the provided docker
# docker run --name nvidia_24_07  --privileged --net=host --ipc=host --gpus=all -v /mnt:/mnt -v /tmp:/tmp -d nvcr.io/nvidia/pytorch:24.07-py3 sleep infinity

cd BitVLA
pip install -e openvla-oft/
pip install -e transformers

cd openvla-oft/

# install LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO/
# in BitVLA
pip install -r experiments/robot/libero/libero_requirements.txt

# install bitvla
pip install -e bitvla/
```

We adopt the same dataset as OpenVLA-OFT for the fine-tuning on LIBERO. You can download the dataset from [HuggingFace](https://huggingface.co/datasets/openvla/modified_libero_rlds).

```
git clone git@hf.co:datasets/openvla/modified_libero_rlds
```

#### 2. OFT fine-tuning

First convert the [BitVLA](https://huggingface.co/hongyuw/bitvla-bitsiglipL-224px-bf16) to a format compatible with the VLA codebase.

```
python convert_ckpt.py /path/to/bitvla-bitsiglipL-224px-bf16
```

After that, you can finetune the BitVLA using the following command. Here we take LIBERO spatial as an example:

```
torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune_bitnet.py \
  --vla_path /path/to/bitvla-bitsiglipL-224px-bf16 \
  --data_root_dir /path/to/modified_libero_rlds/ \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /path/to/save/your/ckpt \
  --use_l1_regression True \
  --warmup_steps 375 \
  --use_lora False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 2 \
  --grad_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --max_steps 10001 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --run_id_note your_id
```

### Evaluation on LIBERO

You can download our fine-tuned BitVLA models from [HuggingFace](https://huggingface.co/collections/hongyuw/bitvla-68468fb1e3aae15dd8a4e36e). As an example for spatial set in LIBERO, run the following script for evaluation:

```
python experiments/robot/libero/run_libero_eval_bitnet.py \
    --pretrained_checkpoint  /path/to/ft-bitvla-bitsiglipL-224px-libero_spatial-bf16 \
    --task_suite_name libero_spatial \
    --info_in_path "information you want to show in path" \
    --model_family "bitnet" 
```

## Acknowledgement

This repository is built using [LMM-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), [the HuggingFace's transformers](https://github.com/huggingface/transformers) and [OpenVLA-OFT](https://github.com/moojink/openvla-oft).

## Citation

If you find this repository useful, please consider citing our work:
```
@article{bitvla,
  title={BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation}, 
  author={Hongyu Wang and Chuyan Xiong and Ruiping Wang and Xilin Chen},
  year={2025},
  eprint={2506.07530},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
}
```

## License
This project is licensed under the MIT License.

### Contact Information

For help or issues using models, please submit a GitHub issue.
