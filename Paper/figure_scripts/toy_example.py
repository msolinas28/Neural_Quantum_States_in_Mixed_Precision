import numpy as np
import matplotlib.pyplot as plt

def tv_bound_sigma(eps, alpha, beta):
    prefactor = 1 / (1 - beta / alpha)
    return prefactor * (1.0 - np.exp(-np.abs(eps)))

def exact_tv(eps, alpha, beta):
    return (beta*np.abs(1-np.exp(eps))) / (alpha + beta*np.exp(eps))

def main():
    # We'll fix alpha = 1 and vary the ratio beta/alpha = rho in (0,1)
    alpha = 1.0
    rhos = [0.01, 0.1, 0.5, 0.9]  # beta/alpha, all < 1 so alpha > beta

    # Sigma range
    sigma_min = -3
    sigma_max = 1
    num_pts = 500

    eps = np.logspace(sigma_min, sigma_max, num_pts)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 18,
        "font.size": 18,
        "legend.fontsize": 12,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    })
    fig, ax_left = plt.subplots(
        1, 1, figsize=(5, 5),
    )
    blues = plt.get_cmap('Set1')
    for i, rho in enumerate(rhos):
        beta = rho * alpha
        bounds = tv_bound_sigma(eps, alpha, beta)
        col = blues((i + 1) / 10)
        ax_left.plot(eps, bounds, label=rf"$\beta/\alpha = {rho}$", color=col)
        ax_left.plot(eps, list(map(lambda x: exact_tv(x, alpha,beta), eps)), linestyle='dashed',
                     color=col)
    ax_left.set_xlabel(r"$\varepsilon$")
    ax_left.set_ylabel(r"TV bound")
    ax_left.set_xscale('log')
    ax_left.set_yscale('log')
    ax_left.grid(True, linestyle="--", alpha=0.5)
    ax_left.legend(fontsize=13)

    indices = np.arange(len(rhos))
    width = 0.35  # width of each bar within a group

    pi0_vals = [1.0 / (1.0 + rho) for rho in rhos]
    pi1_vals = [rho / (1.0 + rho) for rho in rhos]

    greens = plt.get_cmap('Greens')
    reds = plt.get_cmap('Reds')
    c = 0.5
    # ax_right.bar(indices - width / 2, pi0_vals, width=width, label=r"$\pi(0)$", color=greens(c))
    # ax_right.bar(indices + width / 2, pi1_vals, width=width, label=r"$\pi(1)$", color=reds(c))
    #
    # ax_right.set_xticks(indices)
    # ax_right.set_xticklabels([f"{rho:.2f}" for rho in rhos])
    # ax_right.set_xlabel(r"$\beta/\alpha$")
    # ax_right.set_ylabel("Probability")
    # ax_right.set_ylim(0, 1.0)
    # ax_right.legend(fontsize=15)

    plt.tight_layout()
    plt.savefig("figures/toy_example.pdf")
    plt.show()


if __name__ == "__main__":
    main()
