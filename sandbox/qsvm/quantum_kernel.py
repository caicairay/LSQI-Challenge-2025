import numpy as np
import pennylane as qml


# ---------------------------
#  Quantum kernel design (PennyLane) + kernel matrix computation
# ---------------------------
# We'll implement a small angle-encoding feature map on n_qubits qubits.
# Feature map: for each input feature xi, apply RX(xi * scale) on a qubit and entangle with CNOTs.
# Kernel K(x, x') = |<phi(x)|phi(x')>|^2 where |phi(x)> is state prepared by circuit.
class QuantumKernel:
    def __init__(self, n_qubits, qpu="lightning.qubit"):
        self.n_qubits = n_qubits
        self.dev = qml.device(qpu, wires=n_qubits)

        @qml.qnode(self.dev)
        def feature_state_circuit(angles):
            # prepare parameterized state from angles (length = n_qubits)
            for i in range(self.n_qubits):
                qml.RX(angles[i], wires=i)
            # simple entangling layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # optional second layer
            for i in range(self.n_qubits):
                qml.RY(angles[i] * 0.5, wires=i)
            return qml.state()

        self.feature_state_circuit = feature_state_circuit

    def feature_map_angles(self, x):
        # x: 1D array of covariates (e.g., [BW_norm, COMED])
        # project x to n_qubits angles via linear projection (learned or fixed; here fixed)
        x = np.asarray(x, dtype=float)
        # normalize features to [-pi, pi]
        xnorm = (x - x.mean()) / (x.std() + 1e-6)
        # simple tiling/projection
        angles = np.zeros(self.n_qubits)
        for i in range(self.n_qubits):
            angles[i] = np.sum(np.sin((i + 1) * xnorm))  # deterministic mixing
        # scale into [-pi/2, pi/2]
        angles = np.tanh(angles) * (np.pi / 2)
        return angles

    def compute_state_vector(self, x):
        angles = self.feature_map_angles(x)
        return np.array(self.feature_state_circuit(angles))

    def quantum_kernel_matrix(self, X):
        # X: (n_samples, n_features)
        n = X.shape[0]
        states = [self.compute_state_vector(x) for x in X]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                val = np.vdot(states[i], states[j])  # complex inner product
                kval = np.abs(val) ** 2
                K[i, j] = kval
                K[j, i] = kval
        return K
