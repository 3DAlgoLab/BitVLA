# BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation
[[paper]]() [[model]]() [[code]]()

- June 2025: [BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation]()


## Open Source Plan

- ✅ Paper, Pre-trained VLM and evaluation code.
- 🧭 Fine-tuned VLA models, pre-training and fine-tuning code.
- 🧭 Pre-trained VLA.


## Evaluation on VQA

We use the [LMM-Eval]() toolkit to conduct evaluations on VQA tasks. We provide the [transformers repo]() in which we modify the [modeling_llava.py]() and [modeling_siglip.py]() to support the W1.58-A8 quantization. 

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
## Robotics Finetuning on LIBERO

We open-sourced the code for finetuning and evaluation on LIBERO.

This implementation is based on [OpenVLA-OFT](https://github.com/moojink/openvla-oft/tree/main), so please first set up the environment as required by that project. You can refer to [SETUP.md](https://github.com/moojink/openvla-oft/blob/main/SETUP.md) and [LIBERO.md](https://github.com/moojink/openvla-oft/blob/main/LIBERO.md) for detailed instructions.

```
conda create -n bitvla python=3.10 -y
conda activate bitvla

cd BitVLA/openvla-oft
pip install -e .

# install LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
# in BitVLA/openvla-oft
pip install -r experiments/robot/libero/libero_requirements.txt

# install transformers
pip install -e transformers

# install bitvla
pip install -e bitvla

# install torch, we use 2.5.0 version
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
```

After that, you can download the finetuning dataset.

```
git clone git@hf.co:datasets/openvla/modified_libero_rlds
```

After that, you can convert the trained VLM checkpoint to a format compatible with the VLA codebase.

Note: Remember to change `CKPT_PTH` in convert_ckpt.py

```
python convert_ckpt.py
```

After that, you can finetune the VLM using the following command. Here we take LIBERO spatial as an example:

```
torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune_bitnet.py \
  --vla_path /path/to/your/vlm/ckpt \
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

## Evaluataion on LIBERO

You can download our BitVLA model on HuggingFace

| Task suite     | Model |
| -------------- | ----- |
| LIBERO-Spatial |       |
| LIBERO-Object  |       |
| LIBERO-Goal    |       |
| LIBERO-Long    |       |

As an example for spatial, use the following script for evaluation:

```
python experiments/robot/libero/run_libero_eval_bitnet.py \
    --pretrained_checkpoint  /path/to/vla/ckpt \
    --task_suite_name libero_spatial \
    --info_in_path "information you want to show in path" \
    --model_family "bitnet" 
```

## Acknowledgement

This repository is built using [LMM-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), [the HuggingFace's transformers](https://github.com/huggingface/transformers) and [OpenVLA-OFT](https://github.com/moojink/openvla-oft/tree/main).

## License
This project is licensed under the MIT License.

### Contact Information

For help or issues using models, please submit a GitHub issue.
