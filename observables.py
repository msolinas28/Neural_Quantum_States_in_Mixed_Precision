from netket.operator.spin import sigmax, sigmay, sigmaz
import numpy as np

def sigma(direction: str, hi, site: int):
    if direction == "x":
        return sigmax(hi, site)
    if direction == "y":
        return sigmay(hi, site)
    if direction == "z":
        return sigmaz(hi, site)
    else:
        raise ValueError("Direction must be one of the following options: 'x', 'y' or 'z'")

def compute_mag_vector(v, direction, L, hi):
    mag_vector = []

    for i in range(L):
        mag_operator = sigma(direction, hi, i).to_sparse()
        mag_vector.append(np.einsum("i, i", v.conj(), mag_operator @ v))

    return np.array(mag_vector)

def total_magnetization(direction: str, hi):
    M = 0
    for i in range(hi.size):
        M += sigma(direction, hi, i)

    return M / hi.size 

def fidelity(v, w):
    v_norm = np.einsum("i, i", v.conj(), v)
    w_norm = np.einsum("i, i", w.conj(), w)

    F = np.abs(np.einsum("i, i", v.conj(), w)) ** 2 / (v_norm * w_norm)

    return F.real

def entanglement_entropy(v, L):
    L_A = L // 2
    L_B = L - L_A
    dim_A = 2 ** L_A
    dim_B = 2 ** L_B

    v_matrix = v.reshape((dim_A, dim_B))

    rho_A = np.dot(v_matrix, v_matrix.T.conj())  

    rho_A_eigenvalues = np.linalg.eigvalsh(rho_A)
    rho_A_eigenvalues = rho_A_eigenvalues[rho_A_eigenvalues > 1e-10]  

    S = -np.sum(rho_A_eigenvalues * np.log(rho_A_eigenvalues))

    return S

def sample_all_correlation_herm(vstate, direction, hi, g):

    expect = lambda x: vstate.expect(x)

    N = g.n_nodes
    dist_matrix = g.distances() 

    sigmas = [sigma(direction, hi, i) for i in range(N)]

    one_point = np.array([expect(s) for s in sigmas])
    one_point_mean = np.array([op.mean for op in one_point])
    one_point_error = np.array([op.error_of_mean for op in one_point])

    two_point_mean = np.empty((N, N), dtype=complex)
    two_point_error = np.empty((N, N), dtype=complex)
    for i in range(N):
        s_i = sigmas[i]
        for j in range(N):
            s_j = sigmas[j]
            tp = expect(s_i @ s_j)
            two_point_mean[i, j] = tp.mean
            two_point_error[i, j] = tp.error_of_mean

    one_outer_mean = np.outer(one_point_mean, one_point_mean)
    outer_error = np.sqrt(np.outer(one_point_error**2, one_point_mean**2) + np.outer(one_point_mean**2, one_point_error**2))

    connected_mean = two_point_mean - one_outer_mean     
    connected_error = np.sqrt(np.nan_to_num(two_point_error)**2 + outer_error**2)       

    unique_r = np.unique(dist_matrix)
    c_r_mean = []
    cc_r_mean = []
    c_r_error = []
    cc_r_error = []

    for r in unique_r:
        mask = (dist_matrix == r)  

        c_r_mean.append(np.mean(two_point_mean[mask]))
        cc_r_mean.append(np.mean(connected_mean[mask]))

        vals_error = two_point_error[mask]
        c_r_error.append(np.sqrt(np.sum(vals_error**2)) / len(vals_error)) 

        vals_error = connected_error[mask]
        cc_r_error.append(np.sqrt(np.sum(vals_error**2)) / len(vals_error))

    M_mean = np.mean(one_point_mean)
    M_error = np.sqrt(np.sum(one_point_error**2)) / len(one_point_error)

    return M_mean, M_error, c_r_mean, c_r_error, cc_r_mean, cc_r_error