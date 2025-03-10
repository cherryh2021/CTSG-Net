import argparse
import csv
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import pandas as pd
import random
import time
import torch
import torch.optim as optim
import tqdm

from model import STGCN, CTSGNet, GWNT, LSTM, DCRNN, MTGNN, PGCN
import util


def str_to_bool(value):
    """Convert string to boolean value."""
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {'false', 'f', '0', 'no', 'n'}:
        return False
    if value in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f"'{value}' is not a valid boolean value")


def parse_datetime(s):
    """Parse datetime string in 'YYYY-MM-DD HH:MM:SS' format."""
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Not a valid date: '{s}'")


def get_parameters():
    """Parse command-line arguments and set up device configuration."""
    parser = argparse.ArgumentParser(description='Traffic Prediction Model Training')

    # Input settings
    parser.add_argument('--load_path', type=str, default='', help='Path to checkpoint model')
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap', help='Graph shift operator type')
    parser.add_argument('--graph_conv_type', type=str, default='graph_conv', 
                       choices=['cheb_graph_conv', 'graph_conv'], help='Graph convolution type')
    parser.add_argument('--enable_cuda', type=str_to_bool, default=True, help='Enable CUDA (default: True)')
    parser.add_argument('--n_vertex', type=int, default=351, help='Number of vertices in graph')

    # Checkpoint
    parser.add_argument('--checkpoint_epoch', type=int, default=0, help='Starting epoch for checkpoint')

    # Training settings
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for computation')
    parser.add_argument('--cl', type=str_to_bool, default=False, help='Use curriculum learning')
    parser.add_argument('--seq_in_len', type=int, default=20, help='Input sequence length')
    parser.add_argument('--seq_out_len', type=int, default=6, help='Output sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='Weight decay rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=101, help='Random seed')
    parser.add_argument('--clip', type=int, default=5, help='Gradient clipping value')
    parser.add_argument('--model', type=str, default='CTSGNet',
                       choices=['CTSGNet', 'STGCN', 'GWNT', 'LSTM', 'DCRNN', 'MTGNN', 'PGCN'],
                       help='Model type')
    parser.add_argument('--opt', type=str, default='adamw', help='Optimizer type')
    parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate decay factor')
    parser.add_argument('--step_size', type=int, default=15, help='Steps before learning rate decay')
    parser.add_argument('--weighted_lf', type=str_to_bool, default=True, 
                       help='Use weighted loss function')

    # STGCN settings
    parser.add_argument('--n_his', type=int, default=12, help='Historical time steps')
    parser.add_argument('--n_pred', type=int, default=6, help='Prediction time steps')
    parser.add_argument('--time_intvl', type=int, default=5, help='Time interval')
    parser.add_argument('--Kt', type=int, default=3, help='Temporal convolution kernel size')
    parser.add_argument('--stblock_num', type=int, default=2, help='Number of ST blocks')
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'],
                       help='Activation function')
    parser.add_argument('--Ks', type=int, default=3, choices=[3, 2], help='Spatial kernel size')
    parser.add_argument('--enable_bias', type=str_to_bool, default=True, help='Enable bias')
    parser.add_argument('--droprate', type=float, default=0.7, help='Dropout rate')

    # CTSG-Net settings
    parser.add_argument('--enable_signal', type=str_to_bool, default=True, 
                       help='Enable signal processing')
    parser.add_argument('--adap_only', type=str_to_bool, default=False, 
                       help='Use only adaptive matrix')
    parser.add_argument('--dynamic_bool', type=str_to_bool, default=True, 
                       help='Add dynamic matrix')
    parser.add_argument('--buildA_true', type=str_to_bool, default=True, 
                       help='Construct adaptive adjacency matrix')
    parser.add_argument('--gcn_true', type=str_to_bool, default=True, 
                       help='Use graph convolution')
    parser.add_argument('--gcn_type', type=str, default='gated_attention',
                       choices=['gated_attention', 'gated', 'attention', 'mix-hop'],
                       help='GCN type')

    # Result logging
    parser.add_argument('--target_hor', type=int, default=0, help='Target prediction horizon')
    parser.add_argument('--save', type=str, default='garage/', help='Save path')
    parser.add_argument('--print_every', type=int, default=50, help='Print frequency')

    args = parser.parse_args()
    
    # Model-specific adjustments
    if args.model != 'CTSGNet':
        args.enable_signal = False
    args.dataset = 'data//DATA' if args.enable_signal else 'data/DATA_noSignal'
    args.adj_data = 'data/adjacency.pkl'

    print(f'Training configs: {args}')
    set_env(args.seed)
    
    # Device setup
    device = torch.device(args.device) if args.enable_cuda and torch.cuda.is_available() else torch.device('cpu')
    return args, device


def set_env(seed):
    """Set random seeds and environment parameters."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(3)


def data_preparate(args, device):
    """Prepare graph and data loaders."""
    # Load adjacency matrix
    adj = util.load_adj(args.adj_data)
    
    # Model-specific graph processing
    if args.model in ['CTSGNet', 'DCRNN', 'MTGNN', 'PGCN']:
        args.gso = torch.tensor(adj).to(device)
    elif args.model == 'STGCN':
        gso = util.calc_gso(adj, args.gso_type, args.n_vertex)
        if args.graph_conv_type == 'cheb_graph_conv':
            gso = util.calc_chebynet_gso(gso)
        args.gso = torch.from_numpy(gso.toarray().astype(np.float32)).to(device)
    elif args.model == 'GWNT':
        args.gso_type = 'scalap'
        adj = util.adjtype_specification(adj, args.gso_type)
        args.gso = [torch.tensor(i).to(device) for i in adj]
        if False:  # aptonly flag
            args.gso = None
    
    # Load dataset
    dataloader = util.load_dataset(args.dataset, args.batch_size, args.batch_size, args.batch_size)
    return dataloader


def prepare_model(args, device, tanhalpha, hop, gamma):
    """Initialize model, optimizer, and scheduler based on model type."""
    if args.model == 'STGCN':
        Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num
        blocks = [[1]] + [[64, 16, 64] for _ in range(args.stblock_num)]
        blocks.append([128] if Ko == 0 else [128, 128])
        blocks.append([args.seq_out_len])
        model_cls = STGCN.STGCNChebGraphConv if args.graph_conv_type == 'cheb_graph_conv' else STGCN.STGCNGraphConv
        model = model_cls(args, blocks, args.n_vertex)
    
    elif args.model == 'CTSGNet':
        common_params = dict(
            seq_in=args.seq_in_len,
            gcn_true=args.gcn_true,
            buildA_true=args.buildA_true,
            gcn_depth=hop,
            num_nodes=args.n_vertex,
            device=device,
            cycle_num=3,
            predefined_A=args.gso,
            dropout=args.droprate,
            subgraph_size=20,
            node_dim=40,
            dilation_exponential=2,
            conv_channels=32,
            residual_channels=32,
            skip_channels=32,
            end_channels=128,
            seq_length=args.seq_in_len,
            in_dim=1,
            out_dim=args.seq_out_len,
            layers=2,
            tanhalpha=tanhalpha,
            layer_norm_affline=True,
            dynamic_bool=args.dynamic_bool,
            adap_only=args.adap_only,
            gamma=gamma
        )
        
        if not any([args.dynamic_bool, args.buildA_true, args.gcn_true]):
            print("Ablation study")
            model = CTSGNet.gtnet_Signal2(mlp_indim=args.seq_in_len+4, **common_params)
        elif args.enable_signal:
            print("Complete Model")
            model = CTSGNet.gtnet_Signal(gcn_type=args.gcn_type, mlp_indim=args.seq_in_len+4, **common_params)
        else:
            print("Ablation study w/o Signal")
            model = CTSGNet.gtnet(mlp_indim=args.seq_in_len, **common_params)
    
    elif args.model == 'LSTM':
        model = LSTM.VertexLSTM(1, 5, args.seq_out_len, 1, args.droprate)
    
    elif args.model == 'GWNT':
        adjinit = None if True else args.gso[0]  # randomadj flag
        model = GWNT.gwnet(device, args.n_vertex, args.droprate, args.gso, True, True, adjinit,
                         1, args.seq_out_len, 10, 10, 80, 160)
    
    elif args.model == 'DCRNN':
        args.graph_conv_type = "laplacian"
        model = DCRNN.DCRNNModel(device, args.gso, cl_decay_steps=2000, filter_type=args.graph_conv_type,
                               horizon=args.seq_out_len, input_dim=1, l1_decay=0, max_diffusion_step=2,
                               num_nodes=args.n_vertex, num_rnn_layers=2, output_dim=1,
                               rnn_units=64, seq_len=args.seq_in_len, use_curriculum_learning=args.cl)
    
    elif args.model == 'MTGNN':
        model = MTGNN.gtnet(True, False, 1, args.n_vertex, device, args.gso, args.droprate, 40, 5, 1,
                          32, 32, 32, 128, args.seq_in_len, 1, args.seq_out_len, 1, 0.05, 3, True)
    
    elif args.model == 'PGCN':
        model = PGCN.gwnet(args.seq_in_len, args.batch_size, args.device, args.droprate, args.gso,
                         True, True, 1, args.seq_out_len, 12, 12, 96, 192)

    model = model.to(device)
    n_params = sum(p.nelement() for p in model.parameters())
    print(f'Training Model: {args.model}, Number of parameters: {n_params}')
    args.nParams = n_params

    # Optimizer setup
    optimizers = {
        'rmsprop': optim.RMSprop,
        'adam': optim.Adam,
        'adamw': optim.AdamW
    }
    if args.opt not in optimizers:
        raise NotImplementedError(f'Optimizer {args.opt} is not implemented.')
    optimizer = optimizers[args.opt](model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    return model, optimizer, scheduler


def setup_graph(dataloader, model):
    """Setup graph structure for DCRNN model."""
    with torch.no_grad():
        model.eval()
        for x, _ in dataloader['train_loader'].get_iterator():
            trainx = torch.Tensor(x).to(device).permute(0, 3, 1, 2)
            _ = model(trainx)
            break


def train(args, optimizer, scheduler, model, dataloader, device, w0=1, c1=0, c2=1):
    his_val_nonzeroMAE = []
    val_time = []
    train_time = []
    # checkpoint
    if args.checkpoint_epoch > 0:
        print("load model from", args.load_path)
        if args.model == 'DCRNN':
            setup_graph(dataloader, model)
        model.load_state_dict(torch.load(args.save + args.load_path, map_location=lambda storage, loc: storage.cuda(0)))
        previous_best = float(args.load_path.split('_')[4].split('.')[0]+'.'+args.load_path.split('_')[4].split('.')[1])
        print("load best", previous_best)
    print("Start training...")
    save_path = "_epoch_" + args.model + '_0.pth'
    torch.save(model.state_dict(), args.save + save_path)
    print("save to", args.save + save_path)
    train_loss_epoch = list()
    val_loss_epoch = list()
    val_mae_all_epoch = list()
    val_mae_non_epoch = list()
    val_mae_large_epoch = list()
    train_mae_all_epoch = list()
    train_mae_non_epoch = list()
    train_mae_large_epoch = list()
    # start training
    for i in range(args.checkpoint_epoch, args.epochs):
        train_mae_all = []
        train_rmse_all = []
        train_nonzero_mae = []
        train_nonzero_rmse = []
        train_large_mae = []
        train_large_rmse = []
        train_loss = []
        valid_loss = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            # train
            model.train()
            optimizer.zero_grad()
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.permute(0, 3, 1, 2)
            output = model(trainx)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            trainy = trainy[:, 0, :, :]
            real = torch.unsqueeze(trainy, dim=1)
            output = output.permute(0, 2, 3, 1)
            predict = dataloader['scaler'].inverse_transform(output)

            loss = util.weighted_loss(predict, real, w0, c1, c2, weighted_set=args.weighted_lf)
            if i != 0:
                loss.backward()

            if args.clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optimizer.step()

            mae_all, rmse_all, mae_nonzero, rmse_nonzero, mae_large, rmse_large = util.metric(predict, real)
            train_mae_all.append(mae_all)
            train_rmse_all.append(rmse_all)
            train_nonzero_mae.append(mae_nonzero)
            train_nonzero_rmse.append(rmse_nonzero)
            train_large_mae.append(mae_large)
            train_large_rmse.append(rmse_large)
            train_loss.append(loss.item())

            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAE(all data): {:.4f}, Train RMSE(all data): {:.4f}, Train MAE(non-zero data): {:.4f}, Train RMSE(non-zero data): {:.4f}, Train MAE(>100m data): {:.4f}, Train RMSE(>100m data): {:.4f}'
                print(log.format(iter, train_loss[-1], train_mae_all[-1], train_rmse_all[-1], train_nonzero_mae[-1], train_nonzero_rmse[-1], train_large_mae[-1], train_large_rmse[-1]), flush=True)
        t2 = time.time()
        train_time.append(t2-t1)

        # validation
        val_mae_all = []
        val_rmse_all = []
        val_nonzero_mae = []
        val_nonzero_rmse = []
        val_large_mae = []
        val_large_rmse = []
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            valx = torch.Tensor(x).to(device)
            valx = valx.permute(0, 3, 1, 2)
            valy = torch.Tensor(y).to(device)
            valy = valy.transpose(1, 3)
            valy = valy[:, 0, :, :]
            model.eval()
            s1 = time.time()
            output = model(valx).permute(0, 2, 3, 1)
            s2 = time.time()
            real = torch.unsqueeze(valy, dim=1)
            predict = output
            predict = dataloader['scaler'].inverse_transform(output)
            val_loss = util.weighted_loss(predict, real, w0, c1, c2, weighted_set=args.weighted_lf)
            metrics = util.metric(predict, real)
            val_mae_all.append(metrics[0])
            val_rmse_all.append(metrics[1])
            val_nonzero_mae.append(metrics[2])
            val_nonzero_rmse.append(metrics[3])
            val_large_mae.append(metrics[4])
            val_large_rmse.append(metrics[5])
            valid_loss.append(val_loss.item())
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mae_all = np.mean(train_mae_all)
        mtrain_rmse_all = np.mean(train_rmse_all)
        mtrain_mae_nonzero = np.mean(train_nonzero_mae)
        mtrain_rmse_nonzero = np.mean(train_nonzero_rmse)
        mtrain_mae_large = np.mean(train_large_mae)
        mtrain_rmse_large = np.mean(train_large_rmse)
        mvalid_mae_all = np.mean(val_mae_all)
        mvalid_rmse_all = np.mean(val_rmse_all)
        mvalid_mae_nonzero = np.mean(val_nonzero_mae)
        mvalid_rmse_nonzero = np.mean(val_nonzero_rmse)
        mvalid_mae_large = np.mean(val_large_mae)
        mvalid_rmse_large = np.mean(val_large_rmse)
        mvalid_loss = np.mean(valid_loss)
        # choose val_nonzeroMAE as the early stopping indicator
        his_val_nonzeroMAE.append(mvalid_loss)
        train_loss_epoch.append(mtrain_loss)
        train_mae_all_epoch.append(mtrain_mae_all)
        train_mae_non_epoch.append(mtrain_mae_nonzero)
        train_mae_large_epoch.append(mtrain_mae_large)
        val_mae_all_epoch.append(mvalid_mae_all)
        val_mae_non_epoch.append(mvalid_mae_nonzero)
        val_mae_large_epoch.append(mvalid_mae_large)
        val_loss_epoch.append(mvalid_loss)
        log = 'Epoch: {:03d}, Train Loss: {:.6f}, Valid Loss: {:.6f}, Train MAE(all data): {:.4f}, Train RMSE(all data): {:.4f}, Valid MAE(all data): {:.4f}, Valid RMSE(all data): {:.4f}, Train MAE(non-zero data): {:.4f}, Train RMSE(non-zero data): {:.4f}, Valid MAE(non-zero data): {:.4f}, Valid RMSE(non-zero data): {:.4f}, Train MAE(>100m data): {:.4f}, Train RMSE(>100m data): {:.4f}, Valid MAE(>100m data): {:.4f}, Valid RMSE(>100m data): {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mvalid_loss, mtrain_mae_all, mtrain_rmse_all, mvalid_mae_all, mvalid_rmse_all, mtrain_mae_nonzero, mtrain_rmse_nonzero, mvalid_mae_nonzero, mvalid_rmse_nonzero, mtrain_mae_large, mtrain_rmse_large, mvalid_mae_large, mvalid_rmse_large, (t2 - t1)),flush=True)
        path = "_epoch_"+args.model + '_' + str(i+1)+"_"+str(round(mvalid_loss.item(), 4))+".pth"
        torch.save(model.state_dict(), args.save + path)
        print("save to", path)

    # plot the convergence curve
    with open('train_loss/output_'+args.model+'.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(train_loss_epoch)
        writer.writerow(val_loss_epoch)
        writer.writerow(train_mae_all_epoch)
        writer.writerow(train_mae_non_epoch)
        writer.writerow(train_mae_large_epoch)
        writer.writerow(val_mae_all_epoch)
        writer.writerow(val_mae_non_epoch)
        writer.writerow(val_mae_large_epoch)

    # best model up tp now
    if len(his_val_nonzeroMAE) > 0:
        bestid = np.argmin(his_val_nonzeroMAE)
        criterion = round(his_val_nonzeroMAE[bestid].item(), 4)
    else:
        criterion = 1000
    if args.checkpoint_epoch > 0 and criterion > previous_best:
        print("best model up to now", args.load_path)
    else:
        print("best model up to now", "_epoch_"+ args.model + '_'+str(args.checkpoint_epoch + bestid)+"_"+str(criterion)+".pth")

    # return best model path
    if len(his_val_nonzeroMAE) > 0:
        bestid = np.argmin(his_val_nonzeroMAE)
        criterion = round(his_val_nonzeroMAE[bestid].item(),4)
        print("best model after training", "_epoch_"+ args.model + '_' + str(bestid+1)+"_"+str(criterion)+".pth")
    else:
        criterion = 1000

    if args.checkpoint_epoch > 0:
        if criterion <= previous_best:
            path = "_epoch_"+ args.model + '_'+str(args.checkpoint_epoch + bestid+1)+"_"+str(criterion)+".pth"
            print("The valid loss on best model is", criterion)
            print("best model:", path)
            return path
        else:
            print("The valid loss on best model is", previous_best)
            print("best model:", args.load_path)
            return args.load_path
    else:
        print("The valid loss on best model is", criterion)
        path = "_epoch_"+args.model + '_'+str(bestid+1)+"_"+str(criterion)+".pth"
        print("best model:", path)
        return path


@torch.no_grad()
def test(args, model, device, path, dataloader):
    """Test the model."""
    model.eval()
    print(f"Loaded model from {args.save + path}")
    model.load_state_dict(torch.load(args.save + path))

    # Evaluation across horizons
    metrics = {k: [] for k in ['all_mae', 'all_rmse', 'non_mae', 'non_rmse', 'large_mae', 'large_rmse']}
    inference_times = []
    print_out = {k: [] for k in metrics}
    horizons = [0, 1, 2]

    for horizon in range(args.n_pred):
        horizon_metrics = {k: [] for k in metrics}
        for x, y in dataloader['test_loader'].get_iterator():
            x = torch.Tensor(x).to(device).permute(0, 3, 1, 2)
            y = torch.Tensor(y).to(device).transpose(1, 3)[:, 0, :, :]
            real = y.unsqueeze(1)
            s1 = time.time()
            pred = dataloader['scaler'].inverse_transform(model(x).permute(0, 2, 3, 1))
            s2 = time.time()
            inference_times.append(s2 - s1)
            
            m = util.metric(pred[:, :, :, horizon], real[:, :, :, horizon])
            for i, k in enumerate(horizon_metrics):
                horizon_metrics[k].append(m[i])

        means = {k: np.mean(v) for k, v in horizon_metrics.items()}
        print(f"Horizon: {horizon+1} | MAE(all): {means['all_mae']:.4f} | "
              f"RMSE(all): {means['all_rmse']:.4f} | MAE(non-zero): {means['non_mae']:.4f} | "
              f"RMSE(non-zero): {means['non_rmse']:.4f} | MAE(>100m): {means['large_mae']:.4f} | "
              f"RMSE(>100m): {means['large_rmse']:.4f}")
        
        if horizon in horizons:
            for k in metrics:
                print_out[k].append(means[k])

    # Print summary
    for category, keys in [
        ("all data", ['all_mae', 'all_rmse']),
        ("non-zero data", ['non_mae', 'non_rmse']),
        (">100m data", ['large_mae', 'large_rmse'])
    ]:
        print(f"\n{category}")
        print("5sMAE | 10sMAE | 15sMAE | 5sRMSE | 10sRMSE | 15sRMSE")
        values = [print_out[k][i] for k in keys for i in range(3)]
        print(" | ".join(f"{v:.4f}" for v in values))

    avg_inference_time = sum(inference_times) / len(inference_times)
    print(f"\nTraining finished\nAverage inference time: {avg_inference_time:.4f}")

    # Save results
    base_filename = f'output/{args.model}'
    
    with open(f'{base_filename}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f"{t}s{k.upper().replace('_', '')}" for k in ['all_mae', 'all_rmse', 
                 'non_mae', 'non_rmse', 'large_mae', 'large_rmse'] for t in [5, 10, 15]]
        writer.writerow(header)
        writer.writerow([print_out[k][i] for k in metrics for i in range(3)])

    with open(f'output_horizon/{args.model}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f"{5*(i+1)}s" for i in range(args.seq_out_len)])
        writer.writerow([np.mean(horizon_metrics['non_mae']) for horizon in range(args.seq_out_len)])

    return (print_out['all_mae'][0], print_out['non_mae'][0], print_out['large_mae'][0]), avg_inference_time


if __name__ == "__main__":
    args, device = get_parameters()
    dataloader = data_preparate(args, device)

    # Hyperparameters based on time and weighted loss
    if args.weighted_lf:
        tanhalpha, hop, gamma = 8.244264338008563, 2, 0.27764334153801506
        w0, c1, c2 = (0.1703595426402421, 70.75851137169073, 41.43811499417417)
    else:
        tanhalpha, hop, gamma = (8.244264338008563, 2, 0.27764334153801506)
        w0, c1, c2 = 1, 0, 1

    model, optimizer, scheduler = prepare_model(args, device, tanhalpha, hop, gamma)
    
    t1 = time.time()
    best_path = train(args, optimizer, scheduler, model, dataloader, device, w0, c1, c2)
    _, inference_time = test(args, model, device, best_path, dataloader)
    
    t2 = time.time()
    print(f"Best path: {best_path}")
    print(f"Total time spent: {t2-t1:.4f}")
    
    with open(f'output/results_{args.model}.txt', 'w') as f:
        f.write("total run time,inference time,#model parameter\n")
        f.write(f"{t2-t1},{inference_time},{args.nParams}\n")