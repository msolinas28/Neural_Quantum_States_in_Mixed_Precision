# Mixed Precision Neural Quantum States

This repository contains all the data and scripts required to reproduce the results presented in our paper on mixed-precision neural quantum states. Pre-trained models are included, so it is possible to reproduce the figures without running the training procedures.

## Reproducing Paper Figures

All figures can be reproduced using the Jupyter notebooks located in the `Paper/Reproduce/` directory. Each notebook contains detailed instructions and visualization code. Follow the steps below for each figure.

We recommend creating a dedicated virtual environment with Python 3.12 and installing the required Python packages using:
```bash
pip install -r requirements.txt
```

### Figure 1 Panel a

1. Execute sequentially all cells of the notebook [Paper/Reproduce/figure_1_a.ipynb](Paper/Reproduce/figure_1_a.ipynb)

### Figure 2 Panel b

1. Use the following command to run the script and generate the figure
   ```bash
   python Paper/Reproduce/figure_1_b.py --n_samples=2**18 --n=12 --n_seeds=10
   ```
   If this computation is too demanding, you can run the experiment using reduced parameter values:
   ```bash
   python Paper/Reproduce/figure_1_b.py --n_samples=2**14 --n=4 --n_seeds=1
   ```

### Figure 1 Panel c

1. For each value of $h \in \{0.5, 1.0, 1.5, \ldots, 9.5\}$:
   - Train the RBM (repeat this by changing h in --model_params):
     ```bash
     python Script/energy_minimization.py --L=16 --model_params='{"J": 1, "h": 0.5}'
     ```
   - Compute acceptance rates (repeat this by changing h in --h):
     ```bash
     python Script/LPSE/noisy_rbm_acceptance.py --L=16 --h=0.5
     ```
2. The resulting data will be saved in `Data/LPSE/Noisy_rbm/Acceptance/`
3. Open and run the notebook [Paper/Reproduce/figure_1_c.py](Paper/Reproduce/figure_1_c.py) to visualize the data

### Figure 2

1. Train the RBM for TFIM:
   ```bash
   python Script/energy_minimization.py --L=12 --model_params='{"J": 1, "h": 0.5}'
   ```
2. Compute error distributions for the trained RBM:
   ```bash
   python Script/LPSE/eps_fe.py
   ```
3. Compute error distributions for random initialization:
   ```bash
   python Script/LPSE/eps_fe.py --model='random'
   ```
4. Open and run the notebook [Paper/Reproduce/figure_2.ipynb](Paper/Reproduce/figure_2.ipynb) to visualize the data

### Figure 3

1. Train ResCNN models for each $d \in \{\text{None}, \text{f32}, \text{f16}, \text{bf16}\}$ by replacing d in --sampling_dtype=d:
   ```bash
   python Script/LPSO/energy_minimization.py \
       --n_dim=2 \
       --lr=5e-3 \
       --n_chains=$((2**10)) \
       --n_samples=$((2**12)) \
       --M=1000 \
       --save_parameters=True \
       --save_history=True \
       --timeit=True \
       --sampling_dtype=None \
       --L=10 \
       --model='TFIM' \
       --model_params='{"J": 1, "h": 3.04438}' \
       --chunk_size=$((2**11)) \
       --arch='ResCNN' \
       --arch_params='{"n_res_blocks": 4, "filters": 16, "kernel_shape": (3,3), "upcast_sums": False}'
       --compute_sigma=True
   ```
3. Open and run the notebook [Paper/Reproduce/figure_3.ipynb](Paper/Reproduce/figure_3.ipynb) to visualize the data

### Figure 4

1. Run the GPU benchmark script:
   ```bash
   python Script/GPU_test/linear_layer_grid.py
   ```
3. Open and run the notebook [Paper/Reproduce/figure_4.ipynb](Paper/Reproduce/figure_4.ipynb)

Note: Results depend on the specific GPU used (this paper uses NVIDIA H100)

## Repository Structure

- **Script/**: Training and analysis scripts
  - `energy_minimization.py`: Train neural quantum state models
  - `LPSE/`: Scripts for low-precision sampling experiments
  - `LPSO/`: Scripts for low-precision sampling optimization
  - `GPU_test/`: GPU performance benchmarks
- **Data/**: Generated data and results
- **Paper/**: Figure generation scripts and reproduction notebooks
- **Archs/**: Neural network architecture implementations
- **Custom_nk/**: Custom NetKet utilities

## Requirements

See [requirements.txt](requirements.txt) for dependencies.