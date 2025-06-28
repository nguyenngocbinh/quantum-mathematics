# Day 29: Variational Quantum Eigensolver (VQE)

## 🎯 Mục tiêu
- Hiểu nguyên lý hoạt động của VQE
- Triển khai VQE cho ground state estimation
- Áp dụng VQE trong quantum chemistry
- Thiết kế ansatz circuits cho phân tử

## 🧠 VQE - Tổng Quan

### Tại sao VQE?
- **Ground state estimation**: Tìm trạng thái cơ bản của hệ lượng tử
- **Quantum chemistry**: Mô phỏng phân tử và phản ứng hóa học
- **Hybrid quantum-classical**: Kết hợp quantum và classical optimization
- **NISQ compatibility**: Phù hợp với quantum computers hiện tại
- **Energy minimization**: Tối ưu hóa năng lượng của hệ

```python
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SPSA, COBYLA
from qiskit.quantum_info import Pauli
from qiskit.opflow import PauliSumOp, I, Z, X, Y
from qiskit_nature.drivers import PySCFDriver
from qiskit_nature.transformers import ActiveSpaceTransformer
from qiskit_nature.algorithms import GroundStateEigensolver
from qiskit_nature.algorithms import VQEUCCFactory
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
```

## 🔧 VQE Fundamentals

### 1. VQE Algorithm Structure

```python
def vqe_algorithm_overview():
    """
    Tổng quan về thuật toán VQE
    """
    print("VQE Algorithm Steps:")
    print("1. Prepare parameterized quantum state |ψ(θ)⟩")
    print("2. Measure expectation value ⟨ψ(θ)|H|ψ(θ)⟩")
    print("3. Use classical optimizer to minimize energy")
    print("4. Update parameters θ")
    print("5. Repeat until convergence")
    
    return True

# Overview
vqe_algorithm_overview()
```

### 2. Ansatz Circuit Design

```python
def hardware_efficient_ansatz(n_qubits, depth=2):
    """
    Tạo hardware-efficient ansatz circuit
    """
    qc = QuantumCircuit(n_qubits)
    
    # Parameters
    params = []
    param_idx = 0
    
    # Initial layer
    for i in range(n_qubits):
        qc.h(i)
    
    # Ansatz layers
    for layer in range(depth):
        # Single-qubit rotations
        for i in range(n_qubits):
            theta = Parameter(f'θ_{param_idx}')
            params.append(theta)
            qc.ry(theta, i)
            param_idx += 1
        
        # Entangling layer
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        
        # Final layer connects last to first
        if n_qubits > 2:
            qc.cx(n_qubits - 1, 0)
    
    return qc, params

def ucc_ansatz_demo():
    """
    Demo UCC (Unitary Coupled Cluster) ansatz
    """
    # Simple UCC ansatz for 2-qubit system
    qc = QuantumCircuit(2)
    
    # Reference state |01⟩ (Hartree-Fock)
    qc.x(0)
    
    # UCC excitation operator
    theta = Parameter('θ')
    qc.cx(0, 1)
    qc.ry(theta, 1)
    qc.cx(0, 1)
    
    return qc, [theta]

# Test ansatz circuits
n_qubits = 4
hw_ansatz, hw_params = hardware_efficient_ansatz(n_qubits)
ucc_ansatz, ucc_params = ucc_ansatz_demo()

print("Hardware-Efficient Ansatz:")
print(hw_ansatz)
print(f"Number of parameters: {len(hw_params)}")

print("\nUCC Ansatz:")
print(ucc_ansatz)
print(f"Number of parameters: {len(ucc_params)}")
```

## 🧪 VQE Implementation

### 1. Simple VQE cho 2-qubit System

```python
def simple_vqe_demo():
    """
    VQE demo cho hệ 2-qubit đơn giản
    """
    # Hamiltonian: H = Z⊗Z + 0.5*X⊗I
    hamiltonian = PauliSumOp.from_list([
        ('ZZ', 1.0),
        ('XI', 0.5)
    ])
    
    # Ansatz circuit
    qc, params = ucc_ansatz_demo()
    
    # VQE setup
    optimizer = SPSA(maxiter=100)
    backend = Aer.get_backend('statevector_simulator')
    
    # Create VQE
    vqe = VQE(
        ansatz=qc,
        optimizer=optimizer,
        quantum_instance=backend
    )
    
    # Solve
    result = vqe.solve(hamiltonian)
    
    print(f"Ground state energy: {result.optimal_value:.4f}")
    print(f"Optimal parameters: {result.optimal_parameters}")
    
    return result

# Run VQE demo
vqe_result = simple_vqe_demo()
```

### 2. VQE với Custom Ansatz

```python
def custom_ansatz_vqe():
    """
    VQE với custom ansatz cho hệ phức tạp hơn
    """
    # Hamiltonian: H = -Z⊗Z - X⊗X + 0.5*I⊗Z
    hamiltonian = PauliSumOp.from_list([
        ('ZZ', -1.0),
        ('XX', -1.0),
        ('IZ', 0.5)
    ])
    
    # Custom ansatz
    qc = QuantumCircuit(2)
    
    # Parameters
    theta1 = Parameter('θ₁')
    theta2 = Parameter('θ₂')
    theta3 = Parameter('θ₃')
    
    # Ansatz structure
    qc.h(0)
    qc.h(1)
    qc.cx(0, 1)
    qc.ry(theta1, 0)
    qc.ry(theta2, 1)
    qc.cx(0, 1)
    qc.rz(theta3, 0)
    
    # VQE
    optimizer = COBYLA(maxiter=200)
    backend = Aer.get_backend('statevector_simulator')
    
    vqe = VQE(
        ansatz=qc,
        optimizer=optimizer,
        quantum_instance=backend
    )
    
    result = vqe.solve(hamiltonian)
    
    print(f"Ground state energy: {result.optimal_value:.4f}")
    print(f"Optimal parameters: {result.optimal_parameters}")
    
    return result, qc

# Run custom VQE
custom_result, custom_ansatz = custom_ansatz_vqe()
```

## 🧬 Quantum Chemistry với VQE

### 1. Molecular Hamiltonian

```python
def molecular_hamiltonian_demo():
    """
    Demo tạo molecular Hamiltonian
    """
    # H₂ molecule Hamiltonian (simplified)
    # H = -0.011280 * Z⊗Z + 0.171201 * Z⊗I + 0.171201 * I⊗Z + 0.045232 * X⊗X
    
    hamiltonian = PauliSumOp.from_list([
        ('ZZ', -0.011280),
        ('ZI', 0.171201),
        ('IZ', 0.171201),
        ('XX', 0.045232)
    ])
    
    print("H₂ Molecular Hamiltonian:")
    print(hamiltonian)
    
    return hamiltonian

def h2_vqe_simulation():
    """
    VQE simulation cho phân tử H₂
    """
    hamiltonian = molecular_hamiltonian_demo()
    
    # UCC ansatz for H₂
    qc = QuantumCircuit(2)
    
    # Reference state |01⟩ (Hartree-Fock)
    qc.x(0)
    
    # UCC excitation
    theta = Parameter('θ')
    qc.cx(0, 1)
    qc.ry(theta, 1)
    qc.cx(0, 1)
    
    # VQE
    optimizer = SPSA(maxiter=100)
    backend = Aer.get_backend('statevector_simulator')
    
    vqe = VQE(
        ansatz=qc,
        optimizer=optimizer,
        quantum_instance=backend
    )
    
    result = vqe.solve(hamiltonian)
    
    print(f"H₂ Ground State Energy: {result.optimal_value:.6f} Hartree")
    print(f"Optimal parameter: {result.optimal_parameters}")
    
    return result

# H₂ simulation
h2_result = h2_vqe_simulation()
```

### 2. Energy Landscape Analysis

```python
def energy_landscape_analysis():
    """
    Phân tích energy landscape của VQE
    """
    hamiltonian = molecular_hamiltonian_demo()
    
    # Parameter range
    theta_range = np.linspace(0, 2*np.pi, 100)
    energies = []
    
    # UCC ansatz
    qc = QuantumCircuit(2)
    qc.x(0)
    theta = Parameter('θ')
    qc.cx(0, 1)
    qc.ry(theta, 1)
    qc.cx(0, 1)
    
    # Calculate energy for each parameter
    backend = Aer.get_backend('statevector_simulator')
    
    for theta_val in theta_range:
        # Bind parameter
        bound_circuit = qc.bind_parameters({theta: theta_val})
        
        # Execute
        job = execute(bound_circuit, backend)
        statevector = job.result().get_statevector()
        
        # Calculate expectation value
        energy = hamiltonian.eval('statevector', statevector)
        energies.append(energy)
    
    # Plot energy landscape
    plt.figure(figsize=(10, 6))
    plt.plot(theta_range, energies, 'b-', linewidth=2)
    plt.xlabel('Parameter θ')
    plt.ylabel('Energy (Hartree)')
    plt.title('VQE Energy Landscape for H₂')
    plt.grid(True, alpha=0.3)
    
    # Mark minimum
    min_idx = np.argmin(energies)
    plt.plot(theta_range[min_idx], energies[min_idx], 'ro', markersize=10, label='Minimum')
    plt.legend()
    
    plt.show()
    
    print(f"Minimum energy: {min(energies):.6f} Hartree")
    print(f"Optimal parameter: {theta_range[min_idx]:.4f}")
    
    return theta_range, energies

# Energy landscape
theta_vals, energy_vals = energy_landscape_analysis()
```

## 🔬 Advanced VQE Techniques

### 1. Adaptive VQE

```python
def adaptive_vqe_demo():
    """
    Demo adaptive VQE với parameter adaptation
    """
    hamiltonian = molecular_hamiltonian_demo()
    
    # Adaptive ansatz
    qc = QuantumCircuit(2)
    qc.x(0)  # Reference state
    
    # First excitation
    theta1 = Parameter('θ₁')
    qc.cx(0, 1)
    qc.ry(theta1, 1)
    qc.cx(0, 1)
    
    # Second excitation (adaptive)
    theta2 = Parameter('θ₂')
    qc.cx(1, 0)
    qc.ry(theta2, 0)
    qc.cx(1, 0)
    
    # VQE with adaptive optimization
    optimizer = SPSA(maxiter=150)
    backend = Aer.get_backend('statevector_simulator')
    
    vqe = VQE(
        ansatz=qc,
        optimizer=optimizer,
        quantum_instance=backend
    )
    
    result = vqe.solve(hamiltonian)
    
    print(f"Adaptive VQE Energy: {result.optimal_value:.6f} Hartree")
    print(f"Parameters: {result.optimal_parameters}")
    
    return result

# Adaptive VQE
adaptive_result = adaptive_vqe_demo()
```

### 2. VQE với Error Mitigation

```python
def vqe_error_mitigation():
    """
    VQE với error mitigation techniques
    """
    hamiltonian = molecular_hamiltonian_demo()
    
    # Ansatz
    qc, params = ucc_ansatz_demo()
    
    # Multiple backends for comparison
    backends = {
        'statevector': Aer.get_backend('statevector_simulator'),
        'qasm': Aer.get_backend('qasm_simulator', shots=1000)
    }
    
    results = {}
    
    for name, backend in backends.items():
        vqe = VQE(
            ansatz=qc,
            optimizer=SPSA(maxiter=100),
            quantum_instance=backend
        )
        
        result = vqe.solve(hamiltonian)
        results[name] = result
        
        print(f"{name} backend energy: {result.optimal_value:.6f} Hartree")
    
    return results

# Error mitigation demo
error_results = vqe_error_mitigation()
```

## 📊 VQE Performance Analysis

### 1. Convergence Analysis

```python
def vqe_convergence_analysis():
    """
    Phân tích sự hội tụ của VQE
    """
    hamiltonian = molecular_hamiltonian_demo()
    qc, params = ucc_ansatz_demo()
    
    # Track convergence
    energies = []
    iterations = []
    
    def callback(iteration, parameters, mean, std):
        energies.append(mean)
        iterations.append(iteration)
    
    # VQE with callback
    optimizer = SPSA(maxiter=50, callback=callback)
    backend = Aer.get_backend('statevector_simulator')
    
    vqe = VQE(
        ansatz=qc,
        optimizer=optimizer,
        quantum_instance=backend
    )
    
    result = vqe.solve(hamiltonian)
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, energies, 'b-o', linewidth=2, markersize=4)
    plt.xlabel('Iteration')
    plt.ylabel('Energy (Hartree)')
    plt.title('VQE Convergence')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()
    
    print(f"Final energy: {result.optimal_value:.6f} Hartree")
    print(f"Convergence iterations: {len(iterations)}")
    
    return iterations, energies

# Convergence analysis
conv_iterations, conv_energies = vqe_convergence_analysis()
```

## 🎯 Bài tập thực hành

### Bài tập 1: VQE cho Heisenberg Model
```python
def heisenberg_vqe_exercise():
    """
    Bài tập: Implement VQE cho Heisenberg model
    H = J * (X⊗X + Y⊗Y + Z⊗Z)
    """
    # TODO: Implement Heisenberg Hamiltonian
    # TODO: Design appropriate ansatz
    # TODO: Run VQE and find ground state
    pass
```

### Bài tập 2: VQE cho LiH Molecule
```python
def lih_vqe_exercise():
    """
    Bài tập: VQE cho phân tử LiH
    """
    # TODO: Use Qiskit Nature for LiH
    # TODO: Implement UCC ansatz
    # TODO: Compare with classical methods
    pass
```

## 📚 Tài liệu tham khảo

### Sách và Papers:
- "Variational Quantum Eigensolver: A Review of Methods and Best Practices" - Cerezo et al.
- "Quantum Chemistry in the Age of Quantum Computing" - Cao et al.
- Qiskit Nature Documentation

### Công cụ:
- Qiskit Nature
- PennyLane Chemistry
- OpenFermion

### Ứng dụng thực tế:
- Drug discovery
- Material science
- Catalysis research

---

## 🎯 Tổng kết Day 29

### Kỹ năng đạt được:
- ✅ Hiểu nguyên lý VQE và variational principle
- ✅ Thiết kế ansatz circuits cho quantum chemistry
- ✅ Triển khai VQE cho ground state estimation
- ✅ Phân tích energy landscape và convergence
- ✅ Áp dụng VQE cho molecular systems

### Kiến thức quan trọng:
- **Variational principle**: E ≥ ⟨ψ|H|ψ⟩
- **Ansatz design**: Hardware-efficient vs chemistry-inspired
- **Optimization**: Classical optimizers for quantum problems
- **Quantum chemistry**: Molecular Hamiltonians và electronic structure

### Chuẩn bị cho Day 30:
- Neural networks và machine learning concepts
- Parameter optimization techniques
- Quantum-classical hybrid algorithms 