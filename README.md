# Cyclic Temporal-Spatial Graph Convolution Network (CTSG-Net):  a Lane-Level Intersection Queue Length Prediction Model at a Network Scale
## Introduction
We introduces CTSG-Net, a deep learning model for predicting lane-level queue lengths at network-scale intersections. To address the zero-inflated, long-tailed distribution of queue lengths shaped by traffic signal cycles, CTSG-Net employs a cyclic feature extraction module. It also integrates a graph topology learning module, leveraging both static and dynamic adjacency matrices to capture lane-specific spatial dependencies. A multi-range gating attention layer mitigates over-smoothing in graph convolutional networks, while a weighted loss function enhances accuracy, particularly in congestion scenarios. Evaluated on real-world data, CTSG-Net achieves state-of-the-art performance, especially in congestion scenarios.

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
`python3 data_generator.py --add_past_cycle=False --add_signal_index=False`

## Model Training
`python3 main.py --epochs=30 --checkpoint_epoch=0 --device='cuda:0' --load_path='' --seq_in_len=20 --seq_out_len=6 --model='CTSGNet'  --n_vertex=361 --epoch=30`
