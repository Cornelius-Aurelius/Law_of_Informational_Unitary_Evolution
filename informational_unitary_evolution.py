# Informational Unitary Evolution Verification (vOmega Law 38)

import numpy as np

# 1D lattice
N = 200
L = 1.0
x = np.linspace(0.0, L, N, endpoint=False)

# Build Laplacian
dx = x[1] - x[0]
Lmat = np.zeros((N, N), dtype=float)
for i in range(N):
    Lmat[i, i] = -2.0
    Lmat[i, (i - 1) % N] = 1.0
    Lmat[i, (i + 1) % N] = 1.0
Lmat /= dx**2

hbar = 1.0
m = 1.0

# Hamiltonian
V = 5.0 * np.exp(-((x - 0.5*L)**2) / 0.01)
H = -(hbar**2 / (2*m)) * Lmat + np.diag(V)

# Spectral decomposition
eigvals, eigvecs = np.linalg.eigh(H)

def U(dt):
    phase = np.exp(-1j * eigvals * dt / hbar)
    return eigvecs @ (phase[:, None] * eigvecs.conj().T)

# Initial state
psi = np.exp(-((x - 0.25)**2)/0.002) * np.exp(1j * 10*x)
psi = psi / np.sqrt(np.sum(np.abs(psi)**2) * dx)

def norm(psi):
    return np.sum(np.abs(psi)**2) * dx

# Time evolve
dt = 0.01
steps = 300

norms = []

for _ in range(steps):
    norms.append(norm(psi))
    psi = U(dt) @ psi

print("Initial norm:", norms[0])
print("Final norm:", norms[-1])
print("Difference:", norms[-1] - norms[0])
