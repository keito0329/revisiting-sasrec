# Revisiting the Role of Learned Attention Weighting in SASRec

This repository contains the code for our SIGIR 2026 Short Track submission:
**"Revisiting the Role of Learned Attention Weighting in SASRec."**

The project studies when non-uniform attention weighting is functionally necessary in sequential recommendation.
In particular, it compares standard SASRec with a uniform-attention variant implemented via `SFSRec` under matched training settings, and analyzes representation-level positional mixing.


## Repository Layout

- `revisiting_sasrec/`: Main codebase (training, evaluation, analysis scripts, and configs)
- `revisiting_sasrec/runs/`: Hydra entrypoints and experiment configs
- `revisiting_sasrec/src/`: Model and analysis implementation
- `revisiting_sasrec/scripts/`: Utility scripts for plotting and scaling analyses

## Environment Setup

Tested with `python==3.10.16`.

```bash
cd revisiting_sasrec
pip install -r requirements.txt
export SEQ_SPLITS_DATA_PATH=$(pwd)/data
export PYTHONPATH="./"
```

## Data Preparation

Create data directories:

```bash
mkdir -p $SEQ_SPLITS_DATA_PATH/{raw,preprocessed,splitted}
```

Put raw dataset CSV files in:

```bash
$SEQ_SPLITS_DATA_PATH/raw
```

Run preprocessing and splitting (example with Beauty):

```bash
python runs/preprocess.py +dataset=Beauty
python runs/split.py split_type=leave-one-out dataset=Beauty
```

## Training

Run baseline SASRec:

```bash
python runs/train.py model=SASRec dataset=Beauty split_type=leave-one-out
```

Run the uniform-attention variant (implemented as `SFSRec`):

```bash
python runs/train.py model=SFSRec dataset=Beauty split_type=leave-one-out
```

You can switch datasets/split settings using Hydra arguments in `revisiting_sasrec/runs/configs/`.

## Analysis

Representation-level analysis outputs are saved under:

```bash
$SEQ_SPLITS_DATA_PATH/results/analysis/<model>/<dataset>/<split_type>/seed_<seed>/
```

Main analysis/plotting entrypoints:

- `revisiting_sasrec/src/analyze.py`
- `revisiting_sasrec/scripts/compute_best_dropout.py`

