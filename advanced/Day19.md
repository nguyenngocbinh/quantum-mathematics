# Day 19: Quantum Circuit Design Patterns

## 🎯 Mục tiêu
- Hiểu các pattern thiết kế mạch lượng tử phổ biến
- Tạo custom gates và circuit composition
- Tối ưu hóa mạch lượng tử
- Kỹ thuật visualization nâng cao

## 🔧 Circuit Composition Patterns

### 1. Modular Circuit Design

```python
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.quantum_info import Operator
import numpy as np

def create_bell_pair():
    """Tạo Bell pair module"""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    return qc

def create_ghz_module(n_qubits):
    """Tạo GHZ state module"""
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(0)
    for i in range(1, n_qubits):
        qc.cx(0, i)
    return qc

def compose_large_circuit():
    """Compose mạch lớn từ các module nhỏ"""
    qc = QuantumCircuit(4, 4)
    
    # Thêm Bell pair cho qubit 0,1
    bell_01 = create_bell_pair()
    qc = qc.compose(bell_01, qubits=[0, 1])
    
    # Thêm Bell pair cho qubit 2,3
    bell_23 = create_bell_pair()
    qc = qc.compose(bell_23, qubits=[2, 3])
    
    # Entangle tất cả qubit
    qc.cx(1, 2)
    
    qc.measure(range(4), range(4))
    return qc
```

### 2. Parameterized Circuits

```python
from qiskit.circuit import Parameter

def parameterized_rotation(theta):
    """Tạo mạch với tham số"""
    qc = QuantumCircuit(2, 2)
    qc.ry(theta, 0)
    qc.cx(0, 1)
    qc.ry(theta, 1)
    qc.measure([0, 1], [0, 1])
    return qc

def variational_circuit():
    """Variational quantum circuit"""
    theta = Parameter('θ')
    phi = Parameter('φ')
    
    qc = QuantumCircuit(2, 2)
    qc.ry(theta, 0)
    qc.rz(phi, 0)
    qc.cx(0, 1)
    qc.ry(theta, 1)
    qc.measure([0, 1], [0, 1])
    
    return qc

# Bind parameters
def bind_parameters_example():
    qc = variational_circuit()
    bound_circuit = qc.bind_parameters({Parameter('θ'): np.pi/4, Parameter('φ'): np.pi/2})
    return bound_circuit
```

## 🎨 Custom Gates

### 1. Tạo Custom Single-Qubit Gate

```python
def create_custom_gate():
    """Tạo custom gate từ ma trận"""
    # Custom gate: sqrt(X) gate
    sqrt_x_matrix = np.array([[1+1j, 1-1j], [1-1j, 1+1j]]) / 2
    
    from qiskit.extensions import UnitaryGate
    sqrt_x_gate = UnitaryGate(sqrt_x_matrix, label='√X')
    
    qc = QuantumCircuit(1, 1)
    qc.append(sqrt_x_gate, [0])
    qc.measure(0, 0)
    return qc

def create_controlled_custom_gate():
    """Tạo controlled custom gate"""
    # Custom phase gate
    phase_matrix = np.array([[1, 0], [0, np.exp(1j * np.pi/3)]])
    phase_gate = UnitaryGate(phase_matrix, label='P(π/3)')
    
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.append(phase_gate.control(), [0, 1])
    qc.measure([0, 1], [0, 1])
    return qc
```

### 2. Multi-Qubit Custom Gates

```python
def create_swap_like_gate():
    """Tạo gate tương tự SWAP nhưng với phase"""
    # Gate: |00⟩→|00⟩, |01⟩→|10⟩, |10⟩→|01⟩, |11⟩→e^(iπ/4)|11⟩
    matrix = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, np.exp(1j * np.pi/4)]
    ])
    
    custom_2q_gate = UnitaryGate(matrix, label='SWAP+')
    
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.h(1)
    qc.append(custom_2q_gate, [0, 1])
    qc.measure([0, 1], [0, 1])
    return qc
```

## ⚡ Circuit Optimization

### 1. Gate Cancellation

```python
def demonstrate_gate_cancellation():
    """Demonstrate gate cancellation optimization"""
    qc = QuantumCircuit(2, 2)
    
    # Thêm các gate sẽ cancel nhau
    qc.h(0)
    qc.h(0)  # H.H = I
    qc.x(0)
    qc.x(0)  # X.X = I
    
    # Thêm gate thực sự cần thiết
    qc.cx(0, 1)
    
    qc.measure([0, 1], [0, 1])
    return qc

def optimize_circuit(qc):
    """Optimize circuit bằng cách loại bỏ redundant gates"""
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import CommutativeCancellation
    
    pm = PassManager()
    pm.append(CommutativeCancellation())
    optimized_qc = pm.run(qc)
    
    return optimized_qc
```

### 2. Depth Optimization

```python
def create_deep_circuit():
    """Tạo mạch sâu để test optimization"""
    qc = QuantumCircuit(3, 3)
    
    # Thêm nhiều layer operations
    for i in range(5):
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.rz(np.pi/4, 0)
        qc.ry(np.pi/3, 1)
    
    qc.measure(range(3), range(3))
    return qc

def optimize_depth():
    """Optimize circuit depth"""
    qc = create_deep_circuit()
    
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import Depth, Optimize1qGates
    
    pm = PassManager()
    pm.append(Optimize1qGates())
    
    optimized_qc = pm.run(qc)
    
    print(f"Original depth: {qc.depth()}")
    print(f"Optimized depth: {optimized_qc.depth()}")
    
    return optimized_qc
```

## 📊 Advanced Visualization

### 1. Bloch Sphere Visualization

```python
def visualize_bloch_sphere():
    """Visualize qubit states trên Bloch sphere"""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.ry(np.pi/4, 1)
    
    # Không measure để giữ quantum state
    return qc

def plot_circuit_states():
    """Plot states tại các điểm khác nhau trong circuit"""
    qc = QuantumCircuit(1, 1)
    
    # State 1: |0⟩
    qc.h(0)
    
    # State 2: |+⟩
    qc.ry(np.pi/4, 0)
    
    # State 3: Superposition
    qc.rz(np.pi/3, 0)
    
    return qc
```

### 2. Circuit Visualization với Custom Styling

```python
def create_styled_circuit():
    """Tạo circuit với custom styling"""
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.ry(np.pi/3, 0)
    qc.rz(np.pi/4, 1)
    qc.measure([0, 1, 2], [0, 1, 2])
    
    return qc

def plot_with_custom_style():
    """Plot circuit với custom style"""
    qc = create_styled_circuit()
    
    # Custom style
    style = {
        'backgroundcolor': '#002b36',
        'plotter': 'mpl',
        'style': 'iqp',
        'fold': 20
    }
    
    return qc, style
```

## 🔬 Thực hành và Thí nghiệm

### Bài tập 1: Tạo Quantum Fourier Transform Module

```python
def qft_module(n_qubits):
    """Tạo QFT module có thể tái sử dụng"""
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    for i in range(n_qubits):
        qc.h(i)
        for j in range(i+1, n_qubits):
            qc.cp(np.pi/2**(j-i), i, j)
    
    # Swap qubits
    for i in range(n_qubits//2):
        qc.swap(i, n_qubits-1-i)
    
    return qc

def inverse_qft_module(n_qubits):
    """Tạo inverse QFT module"""
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Swap qubits first
    for i in range(n_qubits//2):
        qc.swap(i, n_qubits-1-i)
    
    for i in range(n_qubits-1, -1, -1):
        for j in range(n_qubits-1, i, -1):
            qc.cp(-np.pi/2**(j-i), i, j)
        qc.h(i)
    
    return qc
```

### Bài tập 2: Parameterized Ansatz

```python
def create_ansatz(n_qubits, depth):
    """Tạo parameterized ansatz cho VQE"""
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Tạo parameters
    params = []
    for d in range(depth):
        for i in range(n_qubits):
            theta = Parameter(f'θ_{d}_{i}')
            phi = Parameter(f'φ_{d}_{i}')
            params.extend([theta, phi])
    
    param_idx = 0
    for d in range(depth):
        # Rotation layer
        for i in range(n_qubits):
            qc.ry(params[param_idx], i)
            qc.rz(params[param_idx + 1], i)
            param_idx += 2
        
        # Entanglement layer
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        qc.cx(n_qubits - 1, 0)
    
    return qc
```

### Bài tập 3: Circuit Decomposition

```python
def decompose_to_basis_gates():
    """Decompose custom gate thành basis gates"""
    # Tạo custom gate
    custom_matrix = np.array([
        [np.cos(np.pi/8), -1j*np.sin(np.pi/8)],
        [-1j*np.sin(np.pi/8), np.cos(np.pi/8)]
    ])
    
    custom_gate = UnitaryGate(custom_matrix, label='Custom')
    
    qc = QuantumCircuit(1, 1)
    qc.append(custom_gate, [0])
    
    # Decompose
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import Unroller
    
    pm = PassManager()
    pm.append(Unroller(['u1', 'u2', 'u3', 'cx']))
    decomposed_qc = pm.run(qc)
    
    return qc, decomposed_qc
```

## 🎯 Ứng dụng thực tế

### 1. Quantum Error Correction Circuit

```python
def create_error_correction_circuit():
    """Tạo circuit cho quantum error correction"""
    qc = QuantumCircuit(5, 3)  # 5 qubits, 3 classical bits
    
    # Encode logical qubit
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(0, 3)
    qc.cx(0, 4)
    
    # Syndrome measurement
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(0, 3)
    qc.cx(0, 4)
    qc.measure([1, 2, 3], [0, 1, 2])
    
    return qc
```

### 2. Quantum Teleportation với Custom Gates

```python
def enhanced_teleportation():
    """Quantum teleportation với custom gates"""
    qc = QuantumCircuit(3, 3)
    
    # Prepare state to teleport
    qc.h(0)
    qc.rz(np.pi/6, 0)
    
    # Create Bell pair
    qc.h(1)
    qc.cx(1, 2)
    
    # Teleportation protocol
    qc.cx(0, 1)
    qc.h(0)
    
    # Measure
    qc.measure([0, 1], [0, 1])
    
    # Conditional operations
    qc.cx(1, 2)
    qc.cz(0, 2)
    
    qc.measure(2, 2)
    return qc
```

## 📚 Bài tập về nhà

1. **Custom Gate Library**: Tạo thư viện 5 custom gates hữu ích
2. **Circuit Optimization**: Tối ưu hóa mạch 10 qubit với depth > 50
3. **Parameterized Ansatz**: Thiết kế ansatz cho molecular simulation
4. **Visualization Project**: Tạo interactive circuit visualizer

## 🎯 Kết quả mong đợi
- Thành thạo circuit composition và modular design
- Có thể tạo và sử dụng custom gates
- Hiểu và áp dụng circuit optimization techniques
- Sử dụng advanced visualization tools

## 📖 Tài liệu tham khảo
- [Qiskit Circuit Library](https://qiskit.org/documentation/apidoc/circuit_library.html)
- [Qiskit Transpiler](https://qiskit.org/documentation/apidoc/transpiler.html)
- [Custom Gates in Qiskit](https://qiskit.org/textbook/ch-gates/standard-gates.html) 