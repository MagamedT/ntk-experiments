import torch 
import numpy as np
import matplotlib.pyplot as plt

# Utils to plot learned representation by net vs ntk on grid S^1


def _angle_from_xy(X: torch.Tensor, x_idx: int = 0, y_idx: int = 1) -> np.ndarray:
    """
    Returns angle θ = atan2(y, x) in radians for rows of X (projected onto (x_idx, y_idx)).
    Output in (-π, π]; we sort by θ for plotting.
    """
    x = X[:, x_idx].detach().cpu().numpy()
    y = X[:, y_idx].detach().cpu().numpy()
    return np.arctan2(y, x)

def _aggregate_preds(pred_list, mode="mean", seed_idx=0):
    """
    pred_list: list of 1D tensors (M,). If multiple seeds were stored, aggregate.
    Returns (mu, std) as numpy arrays of shape (M,). std=0 if only one seed.
    """
    if len(pred_list) == 0:
        raise ValueError("Empty prediction list.")
    if mode == "seed":
        mu = pred_list[seed_idx].detach().cpu().numpy().reshape(-1)
        std = np.zeros_like(mu)
        return mu, std
    # default: mean over seeds
    stacked = torch.stack(pred_list, dim=0)  # (S, M)
    mu = stacked.mean(dim=0).detach().cpu().numpy().reshape(-1)
    if stacked.shape[0] > 1:
        std = stacked.std(dim=0, unbiased=True).detach().cpu().numpy().reshape(-1)
    else:
        std = np.zeros_like(mu)
    return mu, std


def plot_ntk_vs_net_by_angle(
    Xte: torch.Tensor,
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    y_ntk_test_dict: dict,
    y_net_test_dict: dict,
    width: int,
    aggregate: str = "mean",   # "mean" across seeds or "seed" to plot a specific seed
    seed_idx: int = 0,
    show_std_band: bool = False,
    x_dim_pair=(0, 1)
):
    """
    Args:
        Xte, Xtr, ytr: train data and grid data.
        y_ntk_test, y_net_test: dicts filled in the training loop with the predictions of NTK and trained network by seed and width.
        width: which width to plot.
        aggregate: "mean" (avg across seeds) or "seed".
        seed_idx: which seed to pick when aggregate == "seed".
        show_std_band: if True and multiple seeds, shows ±1 std band.
        x_dim_pair: which two dims of X to use for angle projection (default (0,1)).
    """
    # angles & sorting for grid
    theta_te = _angle_from_xy(Xte, *x_dim_pair)
    sort_idx = np.argsort(theta_te)
    theta_sorted = theta_te[sort_idx]

    # aggregate predictions across seeds (or pick a seed)
    mu_ntk, sd_ntk = _aggregate_preds(y_ntk_test_dict[width], mode=aggregate, seed_idx=seed_idx)
    mu_net, sd_net = _aggregate_preds(y_net_test_dict[width], mode=aggregate, seed_idx=seed_idx)
    mu_ntk, sd_ntk = mu_ntk[sort_idx], sd_ntk[sort_idx]
    mu_net, sd_net = mu_net[sort_idx], sd_net[sort_idx]

    # training scatter
    theta_tr = _angle_from_xy(Xtr, *x_dim_pair)
    ytr_np = ytr.detach().cpu().numpy().reshape(-1)

    # build the figure
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)

    # Left: NTK predictor
    title_left = f'NTK predictions on $S^1$'
    axs[0].plot(theta_sorted, mu_ntk, lw=2)
    if show_std_band and (sd_ntk is not None) and (sd_ntk.ndim == 1):
        axs[0].fill_between(theta_sorted, mu_ntk - sd_ntk, mu_ntk + sd_ntk, alpha=0.2)
    axs[0].scatter(theta_tr, ytr_np, s=20, alpha=0.85)
    axs[0].set_title(title_left)
    axs[0].set_xlabel(r"angle $\theta$")
    axs[0].set_ylabel(r"$y$")

    # Right: Trained network
    title_right = f"Trained network of width {width} predictions on $S^1$"
    axs[1].plot(theta_sorted, mu_net, lw=2)
    if show_std_band and (sd_net is not None) and (sd_net.ndim == 1):
        axs[1].fill_between(theta_sorted, mu_net - sd_net, mu_net + sd_net, alpha=0.2)
        axs[1].plot(theta_sorted, mu_net, color = "red")
    axs[1].scatter(theta_tr, ytr_np, s=20, alpha=0.85)
    axs[1].set_title(title_right)
    axs[1].set_xlabel(r"angle $\theta$")

    # set the range of plot
    ymins = [ax.get_ylim()[0] for ax in axs]
    ymaxs = [ax.get_ylim()[1] for ax in axs]
    y_min, y_max = min(ymins), max(ymaxs)
    axs[0].set_ylim(y_min, y_max)
    axs[1].set_ylim(y_min, y_max)
    return fig, axs