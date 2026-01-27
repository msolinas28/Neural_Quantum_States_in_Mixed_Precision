import netket as nk
from functools import cached_property
import numpy as np
from scipy.sparse.linalg import eigsh

def Hamiltonian(name, hi, g, use_jax=True, H_par=None, **kwargs):
    """
    Return an Hamiltonian as a local operator.

    Args:
        name: name of the model. 
        H_par: Optional parameters (e.g., {'J': 1.0, 'h': 0.5}).
        kwargs: Optional keyword arguments (e.g., J=1.0, h=0.5).
    Returns:
        List of tuples: [(params, energy), ...]
    """
    
    if use_jax:
        if H_par is None:
            return _Hamiltonian_jax(name, hi, g, **kwargs)
        else:
            return _Hamiltonian_jax(name, hi, g, **H_par)
    else: 
        if H_par is None:
            return _Hamiltonian(name, hi, g, **kwargs)
        else:
            return _Hamiltonian(name, hi, g, **H_par)
    
def _Hamiltonian_jax(name, hi, g, **H_par):
    
    if name == "TFIM":
        return nk.operator.IsingJax(hi, g, **H_par)
    elif name == "Heisenberg":
        return nk.operator.Heisenberg(hilbert = hi, graph = g, sign_rule=True, **H_par).to_jax_operator()
    else:
        raise NotImplementedError

def _Hamiltonian(name, hi, g, **H_par):
    
    if name == "TFIM":
        return nk.operator.Ising(hi, g, **H_par)
    elif name == "Heisenberg":
        return nk.operator.Heisenberg(hilbert = hi, graph = g, sign_rule=True, **H_par)
    else:
        raise NotImplementedError

class System:
    """
    Represents a quantum system defined by a model name, parameters, Hilbert space, and interaction graph.
    Provides methods for constructing the Hamiltonian and computing its eigenvalues and eigenvectors.
    """

    def __init__(self, name: str, H_par: dict | None, hilbert, graph):
        """
        Initialize the system.

        Args:
            name (str): Name of the model (used to construct the Hamiltonian).
            H_par (dict | None): Dictionary of Hamiltonian parameters.
            hilbert: Hilbert space object (typically from NetKet or similar).
            graph: Graph structure defining the system's geometry (e.g. chain, lattice).
        """

        self.name = name
        self.H_par = H_par
        self.hilbert = hilbert
        self.graph = graph

    def __repr__(self):
        return f"System(model = {self.name}, parameters = {self.H_par})"
    
    @cached_property
    def Hamiltonian(self):
        """
        Lazily construct the Hamiltonian using the system's attributes.

        Returns:
            Hamiltonian: A Hamiltonian object generated for this system.
        """

        return Hamiltonian(self.name, self.hilbert, self.graph, True, self.H_par)
    
    @cached_property
    def Hamiltonian_not_jax(self):
        """
        Lazily construct the Hamiltonian using the system's attributes.

        Returns:
            Hamiltonian: A Hamiltonian object generated for this system.
        """

        return Hamiltonian(self.name, self.hilbert, self.graph, False, self.H_par)
    
    @cached_property
    def Hamiltonian_sparse(self):
        """
        Convert the Hamiltonian to sparse matrix format.

        Returns:
            scipy.sparse matrix: Sparse representation of the Hamiltonian.
        """

        return self.Hamiltonian_not_jax.to_sparse()
    
    def eigenvalues(self, k = 2):
        """
        Compute the lowest `k` eigenvalues of the sparse Hamiltonian.

        Args:
            k (int): Number of lowest eigenvalues to compute.

        Returns:
            np.ndarray: Sorted array of the `k` lowest eigenvalues.
        """

        return np.sort(eigsh(self.Hamiltonian_sparse, k = k, which = "SA", return_eigenvectors = False))
    
    def eigenvalues_and_vectors(self, k):
        """
        Compute the lowest `k` eigenvalues and corresponding eigenvectors.

        Args:
            k (int): Number of eigenpairs to compute.

        Returns:
            tuple:
                - np.ndarray: Sorted eigenvalues.
                - np.ndarray: Corresponding eigenvectors as columns.
        """

        vals, vecs = eigsh(self.Hamiltonian_sparse, k=k, which="SA")
        idx = np.argsort(vals)
        return vals[idx], vecs[:, idx]
    
    @cached_property
    def ground_state(self):
        """
        Compute and cache the ground state energy (lowest eigenvalue).

        Returns:
            float: Ground state energy.
        """
        
        return self.eigenvalues()[0]
    
    @property
    def sampler_rule(self):
        if self.name not in ["Heisenberg"]:
            return nk.sampler.rules.LocalRule()
        else: 
            return nk.sampler.rules.ExchangeRule(graph = self.graph)