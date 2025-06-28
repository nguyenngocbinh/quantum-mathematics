# Day 17: Các Cổng Lượng Tử Cơ Bản trong Qiskit

## 🎯 Mục tiêu
- Hiểu và sử dụng các cổng lượng tử cơ bản
- Thực hành với Pauli gates, Hadamard, và Phase gates
- Hiểu ma trận của từng cổng

## 🔧 Các Cổng Lượng Tử Cơ Bản

### 1. Pauli Gates (X, Y, Z)

```python
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import numpy as np

# Pauli-X Gate (NOT gate)
def pauli_x_demo():
    qc = QuantumCircuit(1, 1)
    qc.x(0)  # Áp dụng cổng X
    qc.measure(0, 0)
    return qc

# Pauli-Y Gate
def pauli_y_demo():
    qc = QuantumCircuit(1, 1)
    qc.y(0)  # Áp dụng cổng Y
    qc.measure(0, 0)
    return qc

# Pauli-Z Gate
def pauli_z_demo():
    qc = QuantumCircuit(1, 1)
    qc.h(0)  # Tạo siêu vị trước
    qc.z(0)  # Áp dụng cổng Z
    qc.measure(0, 0)
    return qc
```

### 2. Hadamard Gate (H)

```python
def hadamard_demo():
    qc = QuantumCircuit(1, 1)
    qc.h(0)  # Tạo siêu vị |0⟩ + |1⟩
    qc.measure(0, 0)
    return qc

# Hadamard hai lần = Identity
def hadamard_twice():
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.h(0)  # H² = I
    qc.measure(0, 0)
    return qc
```

### 3. Phase Gates (S, T, P)

```python
def phase_gates_demo():
    qc = QuantumCircuit(1, 1)
    qc.h(0)  # Tạo siêu vị
    qc.s(0)  # Phase gate S (π/2)
    qc.measure(0, 0)
    return qc

def t_gate_demo():
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.t(0)  # T gate (π/4)
    qc.measure(0, 0)
    return qc

def p_gate_demo(phi):
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.p(phi, 0)  # Phase gate với góc tùy ý
    qc.measure(0, 0)
    return qc
```

## 📊 Ma Trận của Các Cổng

```python
import numpy as np

# Ma trận các cổng cơ bản
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]])
T = np.array([[1, 0], [0, np.exp(1j * np.pi/4)]])

print("Ma trận Pauli-X:")
print(X)
print("\nMa trận Hadamard:")
print(H)
```

## 🔬 Thực hành và Thí nghiệm

### Bài tập 1: So sánh các cổng Pauli

```python
def compare_pauli_gates():
    gates = {
        'X': pauli_x_demo(),
        'Y': pauli_y_demo(),
        'Z': pauli_z_demo()
    }
    
    backend = Aer.get_backend('qasm_simulator')
    results = {}
    
    for name, circuit in gates.items():
        job = execute(circuit, backend, shots=1000)
        result = job.result()
        results[name] = result.get_counts(circuit)
        print(f"{name} gate results: {results[name]}")
    
    return results
```

### Bài tập 2: Tạo trạng thái Bell

```python
def bell_state():
    qc = QuantumCircuit(2, 2)
    qc.h(0)  # Hadamard trên qubit đầu
    qc.cx(0, 1)  # CNOT với qubit đầu làm control
    qc.measure([0, 1], [0, 1])
    return qc

# Chạy và phân tích kết quả
bell_circuit = bell_state()
job = execute(bell_circuit, Aer.get_backend('qasm_simulator'), shots=1000)
result = job.result()
print("Bell state results:", result.get_counts(bell_circuit))
```

### Bài tập 3: Khám phá Phase Gates

```python
def phase_exploration():
    angles = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]
    results = {}
    
    for angle in angles:
        qc = p_gate_demo(angle)
        job = execute(qc, Aer.get_backend('qasm_simulator'), shots=1000)
        result = job.result()
        results[f"φ={angle:.2f}"] = result.get_counts(qc)
    
    return results
```

## 🎯 Ứng dụng thực tế

### 1. Tạo trạng thái |+⟩ và |-⟩

```python
def create_plus_minus_states():
    # |+⟩ state
    qc_plus = QuantumCircuit(1, 1)
    qc_plus.h(0)
    qc_plus.measure(0, 0)
    
    # |-⟩ state
    qc_minus = QuantumCircuit(1, 1)
    qc_minus.h(0)
    qc_minus.z(0)  # H + Z = |-
    qc_minus.measure(0, 0)
    
    return qc_plus, qc_minus
```

### 2. Quantum Random Number Generator

```python
def quantum_random_generator():
    qc = QuantumCircuit(1, 1)
    qc.h(0)  # Tạo siêu vị
    qc.measure(0, 0)
    
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=1)
    result = job.result()
    
    return list(result.get_counts(qc).keys())[0]
```

## 📚 Bài tập về nhà

1. **Tạo cổng tùy chỉnh**: Viết hàm tạo cổng quay với góc bất kỳ
2. **Phân tích ma trận**: Tính toán ma trận của H², X², Y², Z²
3. **Trạng thái phức tạp**: Tạo trạng thái |ψ⟩ = (|0⟩ + i|1⟩)/√2
4. **Đo lường**: So sánh kết quả đo của |+⟩ và |-⟩

## 🎯 Kết quả mong đợi
- Hiểu rõ ma trận và tác dụng của từng cổng
- Có thể tạo và phân tích các trạng thái lượng tử cơ bản
- Biết cách sử dụng Qiskit để thực hiện các phép toán lượng tử

## 📖 Tài liệu tham khảo
- [Qiskit Gates Documentation](https://qiskit.org/documentation/stubs/qiskit.circuit.library.html)
- [Quantum Gates Tutorial](https://qiskit.org/textbook/ch-gates/introduction.html) 