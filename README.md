# CTSG-Net
## Introduction
This project introduces CTSG-Net, a novel model designed for fine-grained, network-scale intersection queue length prediction. Key features include:
- Cyclic Feature Extraction Module: Harnesses signal cycle data to model periodic traffic patterns, tackling the zero-inflated and long-tailed nature of queue lengths.
- Graph Topology Learning Module: Combines a static adaptive graph with a dynamic progressive graph learning layer to capture intricate lane-wise spatial relationships.
- Weighted Loss Function: Optimizes performance by addressing the skewed, zero-heavy distribution of queue lengths.

## Overall framework
<img width="790" alt="overall framework" src="https://github.com/user-attachments/assets/9810013a-d44b-43e6-b48f-989867655d3a" />

## Pre-requisites
### Python

This project is created with Python 3.9.18. Virtual environment is highly recommended. It can help you avoid
most dependency issue and reproducible problems.

You can use [Anaconda](https://docs.anaconda.com/anaconda/install/index.html)
or [Miniconda](https://docs.conda.io/en/latest/miniconda.html), or you can use python
standard library [venv](https://docs.python.org/3.8/library/venv.html).

Anaconda/Miniconda will be used in the following documentation.
### Pytorch
This project uses PyTorch 2.1.1. You can install it by following the instructions on the official website [Pytorch](https://pytorch.org). Refer to their documentation for version-specific installation details.
### Requirements
Our code is based on Python3 (>= 3.6). There are a few other dependencies to run the code. The major libraries are listed as follows:

- torch==2.1.1
- pandas==2.0.2
- numpy==1.26.2

## Dataset
Due to data privacy restrictions, we are unable to share the Yizhuang dataset. Instead, we provide a sample dataset generated from a SUMO simulation to demonstrate the model's input data format. You can access it here: [Lane-level Queue Length Dataset Generated from SUMO Simulation](https://github.com/cherryh2021/Lane-level-queue-length-data-in-SUMO-simulation.git)

## Usage

### Get the source code and input data

After uncompress the source code and data, you will have the following directory structure:

```plain
.
├── data                   # Input data for the project.
├── model                  # Code for CTSG-Net and baseline models.
├── README.md
├── data_generator         # Scripts to transform input data into training and testing.
├── main.py                # Main executable script.
├── requirements.txt       # List of Python dependencies
└── utils.py               # Utility functions.
```

### Preparation

#### Create and start the virtual environment

First start the virtual environment:

```shell
$ conda create -n venv python=3.9 -y
$ conda activate venv
(venv) $
```

#### Install dependencies

```shell
(venv) $ pip install -r requirements.txt
```
#### Data Preparation

You can use the `data_generator.py` to generate input data for training and testing process. You can find the detail usage by:

```shell
(venv) $ python data_generator.py --help
                                                                                
usage: data_generator.py [-h] [--dataset DATASET] [--seq_length_x SEQ_LENGTH_X] [--seq_length_y SEQ_LENGTH_Y] [--y_start Y_START] [--dow]
                         [--signal_length SIGNAL_LENGTH] [--cycle_num CYCLE_NUM] [--add_past_cycle ADD_PAST_CYCLE] [--add_signal_index ADD_SIGNAL_INDEX]

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --dataset               Path to the input dataset                            │
│ --seq_length_x          Length of the input sequence for prediction          │
│ --seq_length_y          Length of the output prediction sequence             │
│ --y_start               Starting index for the prediction                    │
│ --dow                   Day of the week                                      │
│ --signal_length         Path to store the signal file                        │
│ --cycle_num             Number of past cycles to include at the same position│
│ --add_past_cycle        Whether to include input from past cycle positions   │
│ --add_signal_index      Whether to include the cycle position index          │
│ --help                  Show this message and exit                           │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### Model Training
```shell
(venv) $ python main.py --help
                                                                                
usage: main.py [-h] [--dataset DATASET] [--adj_data ADJ_DATA] [--load_path LOAD_PATH] [--gso_type GSO_TYPE] [--graph_conv_type {cheb_graph_conv,graph_conv}]
               [--enable_cuda ENABLE_CUDA] [--n_vertex N_VERTEX] [--checkpoint_epoch CHECKPOINT_EPOCH] [--device DEVICE] [--cl CL] [--seq_in_len SEQ_IN_LEN]
               [--seq_out_len SEQ_OUT_LEN] [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY] [--epochs EPOCHS] [--seed SEED]
               [--clip CLIP] [--model {CTSGNet}] [--opt OPT] [--gamma GAMMA] [--step_size STEP_SIZE] [--weighted_lf WEIGHTED_LF] [--droprate DROPRATE]
               [--n_pred N_PRED] [--target_hor TARGET_HOR] [--save SAVE] [--print_every PRINT_EVERY]

╭─ Options ────────────────────────────────────────────────────────────────────────────────────╮
│ -h, --help                  Show this help message and exit                                  │
│ --dataset                   Path to input data                                               │
│ --adj_data                  Path to adjacency matrix                                         │
│ --load_path                 Path to the checkpoint model                                     │
│ --gso_type                  Graph  operator type                                             │
│ --graph_conv_type           Graph convolution type (cheb_graph_conv or graph_conv)           │
│ --enable_cuda               Enable CUDA (default: True)                                      │
│ --n_vertex                  Number of vertices in the graph                                  │
│ --checkpoint_epoch          Starting epoch for the checkpoint                                │
│ --device                    Device for computation (e.g., cpu, cuda)                         │
│ --cl                        Enable curriculum learning                                       │
│ --seq_in_len                Input sequence length                                            │
│ --seq_out_len               Output sequence length                                           │
│ --batch_size                Batch size for training                                          │
│ --learning_rate             Initial learning rate                                            │
│ --weight_decay              Weight decay rate (L2 regularization)                            │
│ --epochs                    Number of training epochs                                        │
│ --seed                      Random seed for reproducibility                                  │
│ --clip                      Gradient clipping value                                          │
│ --model                     Model type (CTSGNet, STGCN, GWNT, LSTM, DCRNN, MTGNN, PGCN)      │
│ --opt                       Optimizer type (e.g., adam, sgd)                                 │
│ --gamma                     Learning rate decay factor                                       │
│ --step_size                 Steps before learning rate decay                                 │
│ --weighted_lf               Enable weighted loss function                                    │
│ --droprate                  Dropout rate                                                     │       
│ --n_pred                    Number of prediction time steps                                  │
│ --target_hor                Target prediction horizon                                        │
│ --save                      Path to save model outputs                                       │
│ --print_every               Frequency of printing training progress                          │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯
```

To get started, use the following command to train the CTSG-Net model for 30 epochs on a GPU with the provided dataset:

`python3 main.py --epochs=30 --checkpoint_epoch=0 --device='cuda:0' --load_path='' --seq_in_len=20 --seq_out_len=6 --n_vertex=361`

### Model Testing
Trained models are saved by default in the garage/ directory. To evaluate the performance of a trained model (e.g., '_epoch_CTSGNet_0.pth'), run the following command:

`python3 main.py --epochs=0 --checkpoint_epoch=1 --device='cuda:0' --load_path='_epoch_CTSGNet_0.pth' --seq_in_len=20 --seq_out_len=6 --n_vertex=361`

Parameters:
- `epochs=0`: Skips training and proceeds directly to testing.
- `checkpoint_epoch=1`: Specifies the starting epoch for loading the checkpoint (any value greater than 0 is acceptable in testing).
- `load_path='_epoch_CTSGNet_0.pth'`: Path to the trained model file in the garage/ directory.
