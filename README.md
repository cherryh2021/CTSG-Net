# CTSG-Net
## Introduction
This project introduces CTSG-Net, a novel model designed for fine-grained, network-scale intersection queue length prediction. Key features include:
- Cyclic Feature Extraction Module: Harnesses signal cycle data to model periodic traffic patterns, tackling the zero-inflated and long-tailed nature of queue lengths.
- Graph Topology Learning Module: Combines a static adaptive graph with a dynamic progressive graph learning layer to capture intricate lane-wise spatial relationships.
- Weighted Loss Function: Optimizes performance by addressing the skewed, zero-heavy distribution of queue lengths.

## Overall framework
<img width="790" alt="overall framework" src="https://github.com/user-attachments/assets/9810013a-d44b-43e6-b48f-989867655d3a" />

## Requirements
Our code is based on Python3 (>= 3.6). There are a few dependencies to run the code. The major libraries are listed as follows:

- torch==2.1.1
- pandas==2.0.2
- numpy==1.26.2
- matplotlib==3.8.2

## Dataset
We regret that, due to data privacy restrictions, we cannot provide the Yizhuang dataset. Instead, we offer a sample dataset generated from a SUMO simulation, available at https://github.com/cherryh2021/Lane-level-queue-length-data-in-SUMO-simulation.git
## Data Preparation
`python3 data_generator.py`

## Model Training
`python3 main.py --epochs=30 --checkpoint_epoch=0 --device='cuda:0' --load_path='' --seq_in_len=20 --seq_out_len=6 --model='CTSGNet'  --n_vertex=361 --epoch=30`
