# Day 20: Quantum Measurement và State Tomography

## 🎯 Mục tiêu
- Hiểu sâu về quantum measurement và POVM
- Thực hiện projective measurement
- State tomography và state reconstruction
- Phân tích lỗi và uncertainty

## 🔍 Quantum Measurement Theory

### 1. Projective Measurement

```python
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Operator, state_fidelity
from qiskit.visualization import plot_histogram, plot_bloch_multivector
import numpy as np

def projective_measurement():
    """Demonstrate projective measurement"""
    qc = QuantumCircuit(2, 2)
    
    # Prepare state |ψ⟩ = (|00⟩ + |11⟩)/√2
    qc.h(0)
    qc.cx(0, 1)
    
    # Projective measurement in computational basis
    qc.measure([0, 1], [0, 1])
    
    return qc

def measurement_in_different_basis():
    """Measure in different bases"""
    qc = QuantumCircuit(1, 1)
    
    # Prepare state |ψ⟩ = |+⟩
    qc.h(0)
    
    # Measure in X basis (Hadamard basis)
    qc.h(0)
    qc.measure(0, 0)
    
    return qc

def measurement_in_y_basis():
    """Measure in Y basis"""
    qc = QuantumCircuit(1, 1)
    
    # Prepare state |ψ⟩ = |+⟩
    qc.h(0)
    
    # Measure in Y basis
    qc.sdg(0)  # S† gate
    qc.h(0)
    qc.measure(0, 0)
    
    return qc
```

### 2. POVM (Positive Operator-Valued Measure)

```python
def povm_measurement():
    """Implement POVM measurement"""
    # Define POVM elements
    # E1 = |0⟩⟨0|, E2 = |1⟩⟨1|, E3 = |+⟩⟨+|
    
    qc = QuantumCircuit(1, 2)  # 2 classical bits for 3 outcomes
    
    # Prepare state |ψ⟩ = |+⟩
    qc.h(0)
    
    # Measure in computational basis (E1, E2)
    qc.measure(0, 0)
    
    # Additional measurement in |+⟩ basis (E3)
    qc.h(0)
    qc.measure(0, 1)
    
    return qc

def create_povm_operators():
    """Create POVM operators"""
    # Example: 3-element POVM
    E1 = np.array([[1, 0], [0, 0]])  # |0⟩⟨0|
    E2 = np.array([[0, 0], [0, 1]])  # |1⟩⟨1|
    E3 = np.array([[0.5, 0.5], [0.5, 0.5]])  # |+⟩⟨+|
    
    # Verify completeness: E1 + E2 + E3 = I
    completeness = E1 + E2 + E3
    print("POVM completeness check:")
    print(completeness)
    
    return E1, E2, E3
```

## 📊 State Tomography

### 1. Single Qubit State Tomography

```python
def single_qubit_tomography():
    """Perform single qubit state tomography"""
    from qiskit.ignis.verification.tomography import state_tomography_circuits
    from qiskit.ignis.verification.tomography import StateTomographyFitter
    
    # Prepare state to be tomographed
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.rz(np.pi/4, 0)  # |ψ⟩ = (|0⟩ + e^(iπ/4)|1⟩)/√2
    
    # Generate tomography circuits
    qst_circs = state_tomography_circuits(qc, [0])
    
    return qc, qst_circs

def analyze_tomography_results():
    """Analyze tomography results"""
    qc, qst_circs = single_qubit_tomography()
    
    # Simulate measurements
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qst_circs, backend, shots=1000)
    result = job.result()
    
    # Fit tomography data
    from qiskit.ignis.verification.tomography import StateTomographyFitter
    qst_fitter = StateTomographyFitter(result, qst_circs)
    rho_fit = qst_fitter.fit()
    
    return rho_fit
```

### 2. Two Qubit State Tomography

```python
def two_qubit_tomography():
    """Perform two qubit state tomography"""
    from qiskit.ignis.verification.tomography import state_tomography_circuits
    
    # Prepare Bell state
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    
    # Generate tomography circuits
    qst_circs = state_tomography_circuits(qc, [0, 1])
    
    return qc, qst_circs

def bell_state_tomography():
    """Tomograph all four Bell states"""
    bell_states = {}
    
    # |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    qc_phi_plus = QuantumCircuit(2, 2)
    qc_phi_plus.h(0)
    qc_phi_plus.cx(0, 1)
    
    # |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
    qc_phi_minus = QuantumCircuit(2, 2)
    qc_phi_minus.h(0)
    qc_phi_minus.cx(0, 1)
    qc_phi_minus.z(0)
    
    # |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
    qc_psi_plus = QuantumCircuit(2, 2)
    qc_psi_plus.h(0)
    qc_psi_plus.cx(0, 1)
    qc_psi_plus.x(1)
    
    # |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
    qc_psi_minus = QuantumCircuit(2, 2)
    qc_psi_minus.h(0)
    qc_psi_minus.cx(0, 1)
    qc_psi_minus.x(1)
    qc_psi_minus.z(0)
    
    states = [qc_phi_plus, qc_phi_minus, qc_psi_plus, qc_psi_minus]
    names = ['|Φ⁺⟩', '|Φ⁻⟩', '|Ψ⁺⟩', '|Ψ⁻⟩']
    
    for name, qc in zip(names, states):
        qst_circs = state_tomography_circuits(qc, [0, 1])
        bell_states[name] = (qc, qst_circs)
    
    return bell_states
```

## 🔬 Advanced Measurement Techniques

### 1. Weak Measurement

```python
def weak_measurement_simulation():
    """Simulate weak measurement"""
    # Weak measurement: partial collapse of wavefunction
    
    qc = QuantumCircuit(2, 2)
    
    # Prepare state |ψ⟩ = |+⟩
    qc.h(0)
    
    # Weak measurement: apply small rotation
    qc.ry(0.1, 0)  # Small rotation angle
    
    # Ancilla qubit for measurement
    qc.cx(0, 1)
    qc.measure(1, 1)  # Measure ancilla
    
    # Final measurement
    qc.measure(0, 0)
    
    return qc

def adaptive_measurement():
    """Adaptive measurement based on previous results"""
    qc = QuantumCircuit(2, 2)
    
    # Prepare state
    qc.h(0)
    qc.ry(np.pi/6, 0)
    
    # First measurement
    qc.measure(0, 0)
    
    # Adaptive second measurement based on first result
    # (This would require classical feedback in real implementation)
    qc.h(0)
    qc.measure(0, 1)
    
    return qc
```

### 2. Continuous Measurement

```python
def continuous_measurement_model():
    """Model continuous measurement"""
    import matplotlib.pyplot as plt
    
    # Simulate continuous measurement of a qubit
    time_steps = 100
    measurement_strength = 0.1
    
    # Initial state: |ψ⟩ = |+⟩
    alpha = 1/np.sqrt(2)  # amplitude of |0⟩
    beta = 1/np.sqrt(2)   # amplitude of |1⟩
    
    # Track state evolution
    alpha_history = [alpha]
    beta_history = [beta]
    
    for t in range(time_steps):
        # Random measurement outcome
        p_0 = abs(alpha)**2
        outcome = np.random.choice([0, 1], p=[p_0, 1-p_0])
        
        # Update state based on measurement
        if outcome == 0:
            alpha = alpha / np.sqrt(p_0)
            beta = beta * np.sqrt(1 - measurement_strength)
        else:
            alpha = alpha * np.sqrt(1 - measurement_strength)
            beta = beta / np.sqrt(1 - p_0)
        
        # Normalize
        norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
        alpha /= norm
        beta /= norm
        
        alpha_history.append(alpha)
        beta_history.append(beta)
    
    return alpha_history, beta_history
```

## 📈 Error Analysis và Uncertainty

### 1. Measurement Error Analysis

```python
def measurement_error_analysis():
    """Analyze measurement errors"""
    qc = QuantumCircuit(1, 1)
    qc.h(0)  # Prepare |+⟩ state
    
    # Ideal measurement should give 50-50 distribution
    qc.measure(0, 0)
    
    # Simulate with different shot counts
    backend = Aer.get_backend('qasm_simulator')
    
    shot_counts = [100, 1000, 10000]
    results = {}
    
    for shots in shot_counts:
        job = execute(qc, backend, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)
        results[shots] = counts
    
    return results

def calculate_measurement_uncertainty():
    """Calculate measurement uncertainty"""
    # For a single qubit measurement
    p = 0.5  # probability of measuring |0⟩
    n = 1000  # number of measurements
    
    # Standard error of the mean
    std_error = np.sqrt(p * (1-p) / n)
    
    # 95% confidence interval
    confidence_interval = 1.96 * std_error
    
    print(f"Standard error: {std_error:.4f}")
    print(f"95% confidence interval: ±{confidence_interval:.4f}")
    
    return std_error, confidence_interval
```

### 2. State Fidelity Calculation

```python
def calculate_state_fidelity():
    """Calculate fidelity between ideal and measured states"""
    from qiskit.quantum_info import state_fidelity
    
    # Ideal Bell state
    ideal_bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
    
    # Simulate noisy Bell state preparation
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    
    # Add some noise (small rotation)
    qc.ry(0.1, 0)
    
    # Get state vector
    backend = Aer.get_backend('statevector_simulator')
    job = execute(qc, backend)
    result = job.result()
    noisy_state = result.get_statevector(qc)
    
    # Calculate fidelity
    fidelity = state_fidelity(ideal_bell, noisy_state)
    
    print(f"State fidelity: {fidelity:.4f}")
    
    return fidelity
```

## 🔬 Thực hành và Thí nghiệm

### Bài tập 1: Complete State Tomography

```python
def complete_tomography_experiment():
    """Complete tomography experiment"""
    # Prepare a complex state
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.ry(np.pi/3, 0)
    qc.cx(0, 1)
    qc.rz(np.pi/4, 1)
    
    # Generate tomography circuits
    from qiskit.ignis.verification.tomography import state_tomography_circuits
    qst_circs = state_tomography_circuits(qc, [0, 1])
    
    # Simulate measurements
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qst_circs, backend, shots=2000)
    result = job.result()
    
    # Fit tomography data
    from qiskit.ignis.verification.tomography import StateTomographyFitter
    qst_fitter = StateTomographyFitter(result, qst_circs)
    rho_fit = qst_fitter.fit()
    
    return qc, rho_fit
```

### Bài tập 2: Measurement in Custom Bases

```python
def custom_basis_measurement():
    """Measure in custom bases"""
    # Define custom basis: |u⟩ = cos(θ)|0⟩ + sin(θ)|1⟩
    #                      |v⟩ = -sin(θ)|0⟩ + cos(θ)|1⟩
    
    theta = np.pi/6
    qc = QuantumCircuit(1, 1)
    
    # Prepare state |ψ⟩ = |+⟩
    qc.h(0)
    
    # Rotate to custom basis
    qc.ry(-theta, 0)
    
    # Measure in computational basis
    qc.measure(0, 0)
    
    return qc

def multiple_basis_measurement():
    """Measure in multiple bases"""
    qc = QuantumCircuit(1, 3)  # 3 classical bits for different bases
    
    # Prepare state
    qc.h(0)
    qc.ry(np.pi/4, 0)
    
    # Measure in Z basis
    qc.measure(0, 0)
    
    # Measure in X basis
    qc.h(0)
    qc.measure(0, 1)
    
    # Measure in Y basis
    qc.sdg(0)
    qc.h(0)
    qc.measure(0, 2)
    
    return qc
```

### Bài tập 3: Error Mitigation

```python
def measurement_error_mitigation():
    """Implement measurement error mitigation"""
    from qiskit.ignis.mitigation.measurement import complete_meas_cal
    
    # Generate calibration circuits
    qr = QuantumRegister(2)
    meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')
    
    # Prepare test state
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    
    return meas_calibs, qc

def apply_error_mitigation():
    """Apply error mitigation to measurement results"""
    meas_calibs, qc = measurement_error_mitigation()
    
    # Simulate with noise
    from qiskit.providers.aer.noise import NoiseModel
    from qiskit.providers.aer.noise.errors import depolarizing_error
    
    # Create noise model
    noise_model = NoiseModel()
    error = depolarizing_error(0.1, 1)
    noise_model.add_all_qubit_quantum_error(error, ['measure'])
    
    # Execute with noise
    backend = Aer.get_backend('qasm_simulator')
    job = execute(meas_calibs + [qc], backend, shots=1000, noise_model=noise_model)
    result = job.result()
    
    # Apply mitigation
    from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
    meas_fitter = CompleteMeasFitter(result, state_labels, circlabel='mcal')
    meas_filter = meas_fitter.filter
    
    # Apply filter to results
    mitigated_result = meas_filter.apply(result)
    
    return result, mitigated_result
```

## 🎯 Ứng dụng thực tế

### 1. Quantum State Estimation

```python
def quantum_state_estimation():
    """Estimate unknown quantum state"""
    # Simulate unknown state preparation
    qc = QuantumCircuit(1, 1)
    
    # Random state parameters
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2*np.pi)
    
    qc.ry(theta, 0)
    qc.rz(phi, 0)
    
    # Perform tomography to estimate state
    from qiskit.ignis.verification.tomography import state_tomography_circuits
    qst_circs = state_tomography_circuits(qc, [0])
    
    return qc, qst_circs, (theta, phi)
```

### 2. Entanglement Witness

```python
def entanglement_witness():
    """Measure entanglement witness"""
    qc = QuantumCircuit(2, 2)
    
    # Prepare potentially entangled state
    qc.h(0)
    qc.cx(0, 1)
    qc.ry(0.3, 0)  # Add some noise
    
    # Measure entanglement witness
    # W = |00⟩⟨00| + |11⟩⟨11| - |01⟩⟨01| - |10⟩⟨10|
    qc.measure([0, 1], [0, 1])
    
    return qc
```

## 📚 Bài tập về nhà

1. **Complete Tomography**: Thực hiện tomography cho 3-qubit GHZ state
2. **Error Analysis**: Phân tích measurement errors trong Bell state
3. **Custom POVM**: Thiết kế và implement POVM cho 3-outcome measurement
4. **Fidelity Tracking**: Theo dõi fidelity theo thời gian với noise

## 🎯 Kết quả mong đợi
- Hiểu sâu về quantum measurement theory
- Thành thạo state tomography techniques
- Có thể phân tích và mitigate measurement errors
- Áp dụng advanced measurement protocols

## 📖 Tài liệu tham khảo
- [Qiskit Ignis Tomography](https://qiskit.org/documentation/ignis/verification/tomography.html)
- [Quantum Measurement Theory](https://qiskit.org/textbook/ch-quantum-hardware/measurement-error-mitigation.html)
- [State Tomography Tutorial](https://qiskit.org/textbook/ch-quantum-hardware/measurement-error-mitigation.html) 