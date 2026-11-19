
import numpy as np


def normalize_state(state):
    """Normalize a quantum state vector."""
    norm = np.linalg.norm(state)

    if norm == 0:
        raise ValueError("State cannot be a zero vector.")

    return state / norm


def calculate_probabilities(state):
    """Calculate measurement probabilities."""
    return np.abs(state) ** 2


def dirac_notation(state):
    """Convert state vector into Dirac notation."""
    num_qubits = int(np.log2(len(state)))
    terms = []

    for i, amplitude in enumerate(state):
        if abs(amplitude) > 1e-6:
            basis = format(i, f"0{num_qubits}b")
            terms.append(
                f"({amplitude.real:.3f}{amplitude.imag:+.3f}j)|{basis}⟩"
            )

    return " + ".join(terms)


def reduced_density_matrix(state, qubit, num_qubits):
    """Calculate one-qubit reduced density matrix."""
    tensor = state.reshape([2] * num_qubits)
    tensor = np.moveaxis(tensor, qubit, 0)
    matrix = tensor.reshape(2, -1)

    return matrix @ matrix.conj().T


def bloch_coordinates(rho):
    """Calculate Bloch sphere coordinates."""
    x = 2 * np.real(rho[0, 1])
    y = -2 * np.imag(rho[0, 1])
    z = np.real(rho[0, 0] - rho[1, 1])

    return float(x), float(y), float(z)


def entanglement_entropy(rho):
    """Calculate von Neumann entropy in bits."""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = np.clip(eigenvalues, 0, 1)

    eigenvalues = eigenvalues[eigenvalues > 1e-12]

    return float(-np.sum(eigenvalues * np.log2(eigenvalues)))
