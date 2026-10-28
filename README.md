# Reinforcing GraphRAG via one latent graph token self-constructed by an RL-guided JEPA architecture.

[![arXiv](https://img.shields.io/badge/arXiv-2402.07630-b31b1b.svg)](https://arxiv.org/abs/2607.00052) 

This repository contains the source code for the paper ["<u>AGE: Adaptive-masking for Graph Embedding in Graph Retrieval-Augmented Generation</u>"](https://arxiv.org/abs/2607.00052).

We introduce **AGE**, We designed the architecture similar to text embedding encoders, addressing the latent feature misalignment in GRAG via a theoretically analyzed JEPA+RL.



## News
We accepted as ECCV 2026 Workshop MARS2! 🎉





## Environment setup
```
conda create --name g_retriever python=3.9 -y
conda activate g_retriever

# https://pytorch.org/get-started/locally/
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

pip install peft
pip install pandas
pip install ogb
pip install transformers
pip install wandb
pip install sentencepiece
pip install torch_geometric
pip install datasets
pip install pcst_fast
pip install gensim
pip install scipy==1.12
pip install protobuf
```

## Download the Llama 2 Model
1. Go to Hugging Face: https://huggingface.co/meta-llama/Llama-2-7b-hf. You will need to share your contact information with Meta to access this model.
2. Sign up for a Hugging Face account (if you don’t already have one).
3. Generate an access token: https://huggingface.co/docs/hub/en/security-tokens.
4. Add your token to the code file as follows:
  ```
  From transformers import AutoModel
  access_token = "hf_..."
  model = AutoModel.from_pretrained("private/model", token=access_token)
  ```


## Dataset
Our work follows G-Retriever, so the dataset comes from https://github.com/XiaoxinHe/G-Retriever/tree/main/dataset.

Please clone the dataset from https://github.com/XiaoxinHe/G-Retriever/tree/main/dataset and place it into the main directory "AGE/dataset" with the name dataset.



## Data Preprocessing
```
# expla_graphs
python -m src.dataset.preprocess.expla_graphs
python -m src.dataset.expla_graphs

# scene_graphs, might take
python -m src.dataset.preprocess.scene_graphs
python -m src.dataset.scene_graphs

# webqsp
python -m src.dataset.preprocess.webqsp
python -m src.dataset.webqsp
```

## Training
Replace path to the llm checkpoints in the `src/model/__init__.py`, then run

### 1) Inference-Only LLM
```
python inference.py --dataset scene_graphs --model_name inference_llm --llm_model_name 7b_chat
```
### 2) Frozen LLM + Prompt Tuning
```
# prompt tuning
python train.py --dataset scene_graphs_baseline --model_name pt_llm

# G-Retriever
python train.py --dataset scene_graphs --model_name graph_llm
```

### 3) Tuned LLM
```
# finetune LLM with LoRA
python train.py --dataset scene_graphs_baseline --model_name llm --llm_frozen False

# G-Retriever with LoRA
python train.py --dataset scene_graphs --model_name graph_llm --llm_frozen False
```

## Citation
```
@article{nguyenhuu2026age,
  title={AGE: Adaptive-masking for Graph Embedding in Graph Retrieval-Augmented Generation},
  author={Nguyen Huu, Bao Long and Hashimoto, Atsushi},
  journal={arXiv preprint arXiv:2607.00052},
  year={2026},
  doi={10.48550/arXiv.2607.00052},
  url={https://arxiv.org/abs/2607.00052}
}
```

## Reproducibility
Use `run.sh` to run the codes and reproduce the published results in the main table.
