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

## Acknowledgement

This repository is built using [LMM-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) and [the HuggingFace's transformers](https://github.com/huggingface/transformers).

## License
This project is licensed under the MIT License.

### Contact Information

For help or issues using models, please submit a GitHub issue.