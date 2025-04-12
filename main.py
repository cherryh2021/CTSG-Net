from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import csv
from datetime import datetime
import os
import time
import numpy as np
import torch
import torch.optim as optim
from model import CTSGNet
import util


def str_to_bool(value):
    """Convert a string to a boolean value."""
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"false", "f", "0", "no", "n"}:
        return False
    if value in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError(f"'{value}' is not a valid boolean value")


def parse_datetime(s):
    """Parse datetime string in 'YYYY-MM-DD HH:MM:SS' format."""
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: '{s}'. Use 'YYYY-MM-DD HH:MM:SS'")


def get_parameters():
    """Parse command-line arguments and configure the device."""
    parser = argparse.ArgumentParser(description="Traffic Prediction Model Training")

    # Input settings
    parser.add_argument("--dataset", type=str, default="data/DATA", help="Path to data")
    parser.add_argument("--adj_data", type=str, default="data/adjacency.pkl", help="Path to adjacency matrix")
    parser.add_argument("--load_path", type=str, default="", help="Path to checkpoint model")
    parser.add_argument("--gso_type", type=str, default="sym_norm_lap", help="Graph shift operator type")
    parser.add_argument("--graph_conv_type", type=str, default="graph_conv", choices=["cheb_graph_conv", "graph_conv"], help="Graph convolution type")
    parser.add_argument("--enable_cuda", type=str_to_bool, default=True, help="Enable CUDA")
    parser.add_argument("--n_vertex", type=int, default=351, help="Number of vertices in graph")

    # Checkpoint
    parser.add_argument("--checkpoint_epoch", type=int, default=0, help="Starting epoch for checkpoint")

    # Training settings
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for computation")
    parser.add_argument("--cl", type=str_to_bool, default=False, help="Use curriculum learning")
    parser.add_argument("--seq_in_len", type=int, default=20, help="Input sequence length")
    parser.add_argument("--seq_out_len", type=int, default=6, help="Output sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.00001, help="Weight decay rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=101, help="Random seed for reproducibility")
    parser.add_argument("--clip", type=int, default=5, help="Gradient clipping value")
    parser.add_argument("--model", type=str, default="CTSGNet", choices=["CTSGNet"], help="Model type")
    parser.add_argument("--opt", type=str, default="adamw", help="Optimizer type (e.g., 'adam', 'adamw')")
    parser.add_argument("--gamma", type=float, default=0.1, help="Learning rate decay factor")
    parser.add_argument("--step_size", type=int, default=15, help="Steps before learning rate decay")
    parser.add_argument("--weighted_lf", type=str_to_bool, default=True, help="Use weighted loss function")
    parser.add_argument("--droprate", type=float, default=0.7, help="Dropout rate")
    parser.add_argument("--n_pred", type=int, default=6, help="Number of prediction time steps")

    # Result logging
    parser.add_argument("--target_hor", type=int, default=0, help="Target prediction horizon")
    parser.add_argument("--save", type=str, default="garage/", help="Directory to save model checkpoints")
    parser.add_argument("--print_every", type=int, default=50, help="Print frequency during training")

    args = parser.parse_args()

    print(f"Training configuration: {args}")
    set_env(args.seed)

    # Device setup
    device = torch.device(args.device) if args.enable_cuda and torch.cuda.is_available() else torch.device("cpu")
    return args, device


def set_env(seed):
    """Set random seeds and environment parameters."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(3)


def data_preparate(args, device):
    """Prepare graph structure and data loaders."""
    # Load adjacency matrix
    adj = util.load_adj(args.adj_data)
    args.gso = torch.tensor(adj).to(device)

    # Load dataset
    dataloader = util.load_dataset(args.dataset, args.batch_size, args.batch_size, args.batch_size)
    return dataloader


def prepare_model(args, device, tanhalpha, hop, gamma):
    """Initialize model, optimizer, and scheduler based on model type."""
    common_params = {
        "seq_in": args.seq_in_len,
        "gcn_depth": hop,
        "num_nodes": args.n_vertex,
        "device": device,
        "cycle_num": 3,
        "predefined_A": args.gso,
        "dropout": args.droprate,
        "seq_length": args.seq_in_len,
        "out_dim": args.seq_out_len,
        "tanhalpha": tanhalpha,
        "gamma": gamma,
    }
    print("CTSGNet Model")
    model = CTSGNet.gtnet_Signal(mlp_indim=args.seq_in_len + 4, **common_params)

    model = model.to(device)
    n_params = sum(p.nelement() for p in model.parameters())
    print(f"Training Model: {args.model}, Number of parameters: {n_params}")
    args.nParams = n_params

    # Optimizer and scheduler setup
    optimizers = {"rmsprop": optim.RMSprop, "adam": optim.Adam, "adamw": optim.AdamW}
    if args.opt not in optimizers:
        raise NotImplementedError(f"Optimizer '{args.opt}' is not supported.")
    optimizer = optimizers[args.opt](model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    return model, optimizer, scheduler


def setup_graph(dataloader, model):
    """Set up graph structure for DCRNN model."""
    with torch.no_grad():
        model.eval()
        for x, _ in dataloader["train_loader"].get_iterator():
            trainx = torch.Tensor(x).to(device).permute(0, 3, 1, 2)
            _ = model(trainx)
            break


def train(args, optimizer, scheduler, model, dataloader, device, w0=1, c1=0, c2=1):
    """Train the model and return the path to the best model."""
    his_val_nonzeroMAE = []
    val_time, train_time = [], []
    train_loss_epoch, val_loss_epoch = [], []
    train_mae_all_epoch, train_mae_non_epoch, train_mae_large_epoch = [], [], []
    val_mae_all_epoch, val_mae_non_epoch, val_mae_large_epoch = [], [], []

    # Load checkpoint if applicable
    if args.checkpoint_epoch > 0:
        print(f"Loading model from {args.load_path}")
        if args.model == "DCRNN":
            setup_graph(dataloader, model)
        model.load_state_dict(torch.load(args.save + args.load_path, map_location=lambda storage, loc: storage.cuda(0)))
        previous_best = float(".".join(args.load_path.split("_")[4].split(".")[:2]))
        print(f"Loaded best validation loss: {previous_best}")

    print("Starting training...")
    initial_save_path = f"_epoch_{args.model}_0.pth"
    torch.save(model.state_dict(), args.save + initial_save_path)
    print(f"Saved initial model to {args.save + initial_save_path}")

    for epoch in range(args.checkpoint_epoch, args.epochs):
        train_metrics = {
            "mae_all": [], "rmse_all": [], "nonzero_mae": [], "nonzero_rmse": [],
            "large_mae": [], "large_rmse": [], "loss": [],
        }
        valid_metrics = {"loss": []}
        t1 = time.time()

        dataloader["train_loader"].shuffle()
        for iter, (x, y) in enumerate(dataloader["train_loader"].get_iterator()):
            model.train()
            optimizer.zero_grad()
            trainx = torch.Tensor(x).to(device).permute(0, 3, 1, 2)
            output = model(trainx)
            trainy = torch.Tensor(y).to(device).transpose(1, 3)[:, 0, :, :]
            real = trainy.unsqueeze(1)
            output = output.permute(0, 2, 3, 1)
            predict = dataloader["scaler"].inverse_transform(output)

            loss = util.weighted_loss(predict, real, w0, c1, c2, weighted_set=args.weighted_lf)
            if epoch != 0:  # Skip backward pass on first epoch if checkpointed
                loss.backward()
                if args.clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()

            mae_all, rmse_all, mae_nonzero, rmse_nonzero, mae_large, rmse_large = util.metric(predict, real)
            train_metrics["mae_all"].append(mae_all)
            train_metrics["rmse_all"].append(rmse_all)
            train_metrics["nonzero_mae"].append(mae_nonzero)
            train_metrics["nonzero_rmse"].append(rmse_nonzero)
            train_metrics["large_mae"].append(mae_large)
            train_metrics["large_rmse"].append(rmse_large)
            train_metrics["loss"].append(loss.item())

            if iter % args.print_every == 0:
                print(
                    f"Iter: {iter:03d}, Train Loss: {loss.item():.4f}, "
                    f"MAE(all): {mae_all:.4f}, RMSE(all): {rmse_all:.4f}, "
                    f"MAE(non-zero): {mae_nonzero:.4f}, RMSE(non-zero): {rmse_nonzero:.4f}, "
                    f"MAE(>100m): {mae_large:.4f}, RMSE(>100m): {rmse_large:.4f}",
                    flush=True,
                )

        t2 = time.time()
        train_time.append(t2 - t1)

        # Validation
        valid_metrics.update({
            "mae_all": [], "rmse_all": [], "nonzero_mae": [], "nonzero_rmse": [],
            "large_mae": [], "large_rmse": [],
        })
        for x, y in dataloader["val_loader"].get_iterator():
            valx = torch.Tensor(x).to(device).permute(0, 3, 1, 2)
            valy = torch.Tensor(y).to(device).transpose(1, 3)[:, 0, :, :]
            real = valy.unsqueeze(1)
            model.eval()
            s1 = time.time()
            output = model(valx).permute(0, 2, 3, 1)
            s2 = time.time()
            predict = dataloader["scaler"].inverse_transform(output)
            val_loss = util.weighted_loss(predict, real, w0, c1, c2, weighted_set=args.weighted_lf)
            metrics = util.metric(predict, real)

            valid_metrics["mae_all"].append(metrics[0])
            valid_metrics["rmse_all"].append(metrics[1])
            valid_metrics["nonzero_mae"].append(metrics[2])
            valid_metrics["nonzero_rmse"].append(metrics[3])
            valid_metrics["large_mae"].append(metrics[4])
            valid_metrics["large_rmse"].append(metrics[5])
            valid_metrics["loss"].append(val_loss.item())

        val_time.append(s2 - s1)
        print(f"Epoch: {epoch:03d}, Inference Time: {s2 - s1:.4f} secs")

        # Compute epoch averages
        train_avg = {k: np.mean(v) for k, v in train_metrics.items()}
        valid_avg = {k: np.mean(v) for k, v in valid_metrics.items()}
        his_val_nonzeroMAE.append(valid_avg["loss"])

        # Store epoch metrics
        train_loss_epoch.append(train_avg["loss"])
        val_loss_epoch.append(valid_avg["loss"])
        train_mae_all_epoch.append(train_avg["mae_all"])
        train_mae_non_epoch.append(train_avg["nonzero_mae"])
        train_mae_large_epoch.append(train_avg["large_mae"])
        val_mae_all_epoch.append(valid_avg["mae_all"])
        val_mae_non_epoch.append(valid_avg["nonzero_mae"])
        val_mae_large_epoch.append(valid_avg["large_mae"])

        print(
            f"Epoch: {epoch:03d}, Train Loss: {train_avg['loss']:.6f}, Valid Loss: {valid_avg['loss']:.6f}, "
            f"Train MAE(all): {train_avg['mae_all']:.4f}, Train RMSE(all): {train_avg['rmse_all']:.4f}, "
            f"Valid MAE(all): {valid_avg['mae_all']:.4f}, Valid RMSE(all): {valid_avg['rmse_all']:.4f}, "
            f"Train MAE(non-zero): {train_avg['nonzero_mae']:.4f}, Train RMSE(non-zero): {train_avg['nonzero_rmse']:.4f}, "
            f"Valid MAE(non-zero): {valid_avg['nonzero_mae']:.4f}, Valid RMSE(non-zero): {valid_avg['nonzero_rmse']:.4f}, "
            f"Train MAE(>100m): {train_avg['large_mae']:.4f}, Train RMSE(>100m): {train_avg['large_rmse']:.4f}, "
            f"Valid MAE(>100m): {valid_avg['large_mae']:.4f}, Valid RMSE(>100m): {valid_avg['large_rmse']:.4f}, "
            f"Training Time: {t2 - t1:.4f}/epoch",
            flush=True,
        )

        # Save model checkpoint
        path = f"_epoch_{args.model}_{epoch + 1}_{round(valid_avg['loss'], 4)}.pth"
        torch.save(model.state_dict(), args.save + path)
        print(f"Saved to {args.save + path}")

        scheduler.step()

    # Determine best model
    best_id = np.argmin(his_val_nonzeroMAE) if his_val_nonzeroMAE else 0
    criterion = round(his_val_nonzeroMAE[best_id], 4) if his_val_nonzeroMAE else 1000
    best_path = f"_epoch_{args.model}_{best_id + 1}_{criterion}.pth"

    if args.checkpoint_epoch > 0 and criterion > previous_best:
        print(f"Best model up to now: {args.load_path} (Previous best: {previous_best})")
        return args.load_path
    print(f"Best model after training: {best_path} (Validation loss: {criterion})")
    return best_path


@torch.no_grad()
def test(args, model, device, path, dataloader):
    """Evaluate the model on the test set."""
    model.eval()
    print(f"Loaded model from {args.save + path}")
    model.load_state_dict(torch.load(args.save + path))

    metrics = ["all_mae", "all_rmse", "non_mae", "non_rmse", "large_mae", "large_rmse"]
    results = {k: [] for k in metrics}
    inference_times = []
    horizons = [0, 1, 2]  # 5s, 10s, 15s

    for horizon in range(args.n_pred):
        horizon_metrics = {k: [] for k in metrics}
        for x, y in dataloader["test_loader"].get_iterator():
            x = torch.Tensor(x).to(device).permute(0, 3, 1, 2)
            y = torch.Tensor(y).to(device).transpose(1, 3)[:, 0, :, :]
            real = y.unsqueeze(1)
            s1 = time.time()
            pred = dataloader["scaler"].inverse_transform(model(x).permute(0, 2, 3, 1))
            s2 = time.time()
            inference_times.append(s2 - s1)

            m = util.metric(pred[:, :, :, horizon], real[:, :, :, horizon])
            for i, k in enumerate(horizon_metrics):
                horizon_metrics[k].append(m[i])

        means = {k: np.mean(v) for k, v in horizon_metrics.items()}
        print(
            f"Horizon: {horizon + 1} | MAE(all): {means['all_mae']:.4f} | "
            f"RMSE(all): {means['all_rmse']:.4f} | MAE(non-zero): {means['non_mae']:.4f} | "
            f"RMSE(non-zero): {means['non_rmse']:.4f} | MAE(>100m): {means['large_mae']:.4f} | "
            f"RMSE(>100m): {means['large_rmse']:.4f}"
        )

        if horizon in horizons:
            for k in metrics:
                results[k].append(means[k])

    # Print summary for specific horizons
    for category, keys in [
        ("All data", ["all_mae", "all_rmse"]),
        ("Non-zero data", ["non_mae", "non_rmse"]),
        (">100m data", ["large_mae", "large_rmse"]),
    ]:
        print(f"\n{category}")
        print("5sMAE | 10sMAE | 15sMAE | 5sRMSE | 10sRMSE | 15sRMSE")
        values = [results[k][i] for k in keys for i in range(3)]
        print(" | ".join(f"{v:.4f}" for v in values))

    avg_inference_time = np.mean(inference_times)
    print(f"\nTraining finished\nAverage inference time: {avg_inference_time:.4f} secs")

    # Save results
    os.makedirs("output", exist_ok=True)
    with open(f"output/{args.model}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"{t}s{k.upper().replace('_', '')}" for k in metrics for t in [5, 10, 15]]
        writer.writerow(header)
        writer.writerow([results[k][i] for k in metrics for i in range(3)])


    return (results["all_mae"][0], results["non_mae"][0], results["large_mae"][0]), avg_inference_time


if __name__ == "__main__":
    args, device = get_parameters()
    dataloader = data_preparate(args, device)

    # Hyperparameters based on weighted loss
    if args.weighted_lf:
        tanhalpha, hop, gamma = 8.24, 2, 0.28
        w0, c1, c2 = 0.17, 70.76, 41.44
    else:
        tanhalpha, hop, gamma = 8.24, 2, 0.28
        w0, c1, c2 = 1, 0, 1

    model, optimizer, scheduler = prepare_model(args, device, tanhalpha, hop, gamma)

    t1 = time.time()
    best_path = train(args, optimizer, scheduler, model, dataloader, device, w0, c1, c2)
    mae_metrics, inference_time = test(args, model, device, best_path, dataloader)
    t2 = time.time()

    print(f"Best model path: {best_path}")
    print(f"Total time spent: {t2 - t1:.4f} secs")

    os.makedirs("output", exist_ok=True)
    with open(f"output/results_{args.model}.txt", "w") as f:
        f.write("Total run time, Inference time, #Model parameters\n")
        f.write(f"{t2 - t1:.4f}, {inference_time:.4f}, {args.nParams}\n")
