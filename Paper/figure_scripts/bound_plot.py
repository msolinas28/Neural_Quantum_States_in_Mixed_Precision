import numpy as np
import matplotlib.pyplot as plt

# Enable LaTeX rendering and serif fonts
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

# Define phi(a) = min(1, a)
def phi(a):
    return np.minimum(1, a)

# Define Δ(s, ε)
def Delta(s, eps):
    return np.abs(phi(s) - phi(s * np.exp(eps)))

def Delta2(s, eps):
    return Delta(s,eps) - s*np.abs(1 -  np.exp(eps))

def Delta3(s, eps):
    return Delta(s,eps) - (1 -  np.exp(-np.abs(eps)))
# Define ranges for s and ε
gran = 3

s_vals = np.logspace(-4, 2, 400)
eps_vals = np.linspace(0., gran, 400)
S, EPS = np.meshgrid(s_vals, eps_vals)

# Compute Δ
D = Delta(S, EPS)

# Plot
min_val = 0
plt.figure(figsize=(7, 5))
levels = np.linspace(min_val, 1, 51)
contour = plt.contourf(S, EPS, D, levels=levels, cmap='coolwarm', vmin=min_val, vmax=1,zorder=-1)
plt.plot(s_vals, list(map(lambda s: -np.log(s), s_vals)), zorder=0, color='black')
plt.vlines(1., -gran, gran, zorder=0, color='black', linestyle='dashed')
cbar = plt.colorbar(contour, label=r'$|\Delta \alpha(x,y)|$', fraction=0.046, pad=0.04)
cbar.set_ticks(np.linspace(min_val, 1, 6))
plt.xlabel(r'$s(x,y)$')
plt.ylabel(r'$\varepsilon (x,y)$')
plt.ylim(0., gran)
plt.xscale('log')
# plt.title(r'$\Delta(s,\varepsilon) = |\phi(s) - \phi(se^{\varepsilon})|,\quad \phi(a)=\min(1,a)$', fontsize=16)
plt.tight_layout()
plt.savefig("figures/alpha.pdf")
plt.show()
