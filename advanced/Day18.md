# Day 18: Cổng Điều Khiển và Đa Qubit

## 🎯 Mục tiêu
- Hiểu và sử dụng các cổng điều khiển (CNOT, CZ, etc.)
- Làm việc với hệ thống đa qubit
- Tạo các trạng thái rối lượng tử phức tạp

## 🔧 Các Cổng Điều Khiển Cơ Bản

### 1. CNOT Gate (Controlled-NOT)

```python
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import numpy as np

def cnot_demo():
    qc = QuantumCircuit(2, 2)
    
    # Tạo trạng thái |10⟩
    qc.x(0)  # Qubit đầu tiên thành |1⟩
    
    # Áp dụng CNOT
    qc.cx(0, 1)  # Control qubit 0, target qubit 1
    
    qc.measure([0, 1], [0, 1])
    return qc

# Thử nghiệm với các trạng thái đầu vào khác nhau
def cnot_truth_table():
    inputs = ['00', '01', '10', '11']
    results = {}
    
    for input_state in inputs:
        qc = QuantumCircuit(2, 2)
        
        # Tạo trạng thái đầu vào
        if input_state[0] == '1':
            qc.x(0)
        if input_state[1] == '1':
            qc.x(1)
            
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        
        backend = Aer.get_backend('qasm_simulator')
        job = execute(qc, backend, shots=1000)
        result = job.result()
        results[input_state] = result.get_counts(qc)
    
    return results
```

### 2. CZ Gate (Controlled-Z)

```python
def cz_demo():
    qc = QuantumCircuit(2, 2)
    
    # Tạo siêu vị trên cả hai qubit
    qc.h(0)
    qc.h(1)
    
    # Áp dụng CZ
    qc.cz(0, 1)
    
    qc.measure([0, 1], [0, 1])
    return qc

def cz_vs_cnot():
    # So sánh tác dụng của CZ và CNOT
    qc_cz = QuantumCircuit(2, 2)
    qc_cz.h(0)
    qc_cz.h(1)
    qc_cz.cz(0, 1)
    qc_cz.measure([0, 1], [0, 1])
    
    qc_cnot = QuantumCircuit(2, 2)
    qc_cnot.h(0)
    qc_cnot.h(1)
    qc_cnot.cx(0, 1)
    qc_cnot.measure([0, 1], [0, 1])
    
    return qc_cz, qc_cnot
```

### 3. CCX Gate (Toffoli - Controlled-Controlled-X)

```python
def toffoli_demo():
    qc = QuantumCircuit(3, 3)
    
    # Tạo trạng thái |110⟩
    qc.x(0)
    qc.x(1)
    
    # Áp dụng Toffoli gate
    qc.ccx(0, 1, 2)  # Control qubits 0,1; target qubit 2
    
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc
```

## 🔗 Tạo Trạng Thái Rối Lượng Tử

### 1. Bell States (4 trạng thái Bell)

```python
def bell_states():
    # |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    def bell_phi_plus():
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        return qc
    
    # |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
    def bell_phi_minus():
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.z(0)
        qc.measure([0, 1], [0, 1])
        return qc
    
    # |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
    def bell_psi_plus():
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.x(1)
        qc.measure([0, 1], [0, 1])
        return qc
    
    # |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
    def bell_psi_minus():
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.x(1)
        qc.z(0)
        qc.measure([0, 1], [0, 1])
        return qc
    
    return {
        '|Φ⁺⟩': bell_phi_plus(),
        '|Φ⁻⟩': bell_phi_minus(),
        '|Ψ⁺⟩': bell_psi_plus(),
        '|Ψ⁻⟩': bell_psi_minus()
    }
```

### 2. GHZ State (Greenberger-Horne-Zeilinger)

```python
def ghz_state(n_qubits=3):
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Tạo siêu vị trên qubit đầu
    qc.h(0)
    
    # Áp dụng CNOT từ qubit đầu đến tất cả qubit khác
    for i in range(1, n_qubits):
        qc.cx(0, i)
    
    qc.measure(range(n_qubits), range(n_qubits))
    return qc

# Tạo GHZ state với 4 qubit
ghz_4 = ghz_state(4)
```

### 3. W State

```python
def w_state(n_qubits=3):
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Tạo W state: (|100⟩ + |010⟩ + |001⟩)/√3
    qc.ry(2*np.arccos(1/np.sqrt(n_qubits)), 0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    
    qc.measure(range(n_qubits), range(n_qubits))
    return qc
```

## 🔬 Thực hành và Thí nghiệm

### Bài tập 1: Phân tích CNOT

```python
def analyze_cnot():
    # Tạo tất cả trạng thái đầu vào có thể
    input_states = ['00', '01', '10', '11']
    results = {}
    
    backend = Aer.get_backend('qasm_simulator')
    
    for state in input_states:
        qc = QuantumCircuit(2, 2)
        
        # Tạo trạng thái đầu vào
        if state[0] == '1':
            qc.x(0)
        if state[1] == '1':
            qc.x(1)
        
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        
        job = execute(qc, backend, shots=1000)
        result = job.result()
        results[state] = result.get_counts(qc)
    
    return results
```

### Bài tập 2: Tạo trạng thái rối tùy chỉnh

```python
def custom_entangled_state(theta, phi):
    qc = QuantumCircuit(2, 2)
    
    # Tạo trạng thái: cos(θ)|00⟩ + e^(iφ)sin(θ)|11⟩
    qc.ry(2*theta, 0)
    qc.cx(0, 1)
    qc.p(phi, 0)
    
    qc.measure([0, 1], [0, 1])
    return qc

# Thử nghiệm với các góc khác nhau
angles = [(np.pi/4, 0), (np.pi/4, np.pi/2), (np.pi/3, np.pi)]
```

### Bài tập 3: Quantum Teleportation Circuit

```python
def quantum_teleportation():
    qc = QuantumCircuit(3, 3)
    
    # Qubit 0: trạng thái cần teleport
    qc.h(0)
    qc.z(0)
    
    # Qubit 1,2: Bell pair
    qc.h(1)
    qc.cx(1, 2)
    
    # Teleportation protocol
    qc.cx(0, 1)
    qc.h(0)
    
    # Đo lường
    qc.measure([0, 1], [0, 1])
    
    # Classical correction
    qc.cx(1, 2)
    qc.cz(0, 2)
    
    qc.measure(2, 2)
    return qc
```

## 🎯 Ứng dụng thực tế

### 1. Quantum Error Correction (3-qubit code)

```python
def three_qubit_code():
    qc = QuantumCircuit(3, 3)
    
    # Encode logical |0⟩
    qc.cx(0, 1)
    qc.cx(0, 2)
    
    # Simulate error (bit flip on qubit 1)
    qc.x(1)
    
    # Syndrome measurement
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.measure([1, 2], [1, 2])
    
    return qc
```

### 2. Quantum Fourier Transform (QFT)

```python
def qft_2_qubit():
    qc = QuantumCircuit(2, 2)
    
    # Apply QFT
    qc.h(0)
    qc.cp(np.pi/2, 0, 1)
    qc.h(1)
    qc.swap(0, 1)
    
    qc.measure([0, 1], [0, 1])
    return qc
```

## 📚 Bài tập về nhà

1. **Tạo cổng điều khiển tùy chỉnh**: Viết hàm tạo controlled-RY gate
2. **Phân tích rối lượng tử**: Tính toán độ rối (entanglement) của Bell states
3. **Quantum Circuit Design**: Thiết kế mạch tạo trạng thái |ψ⟩ = (|000⟩ + |111⟩)/√2
4. **Error Detection**: Tạo mạch phát hiện lỗi bit-flip trên 3 qubit

## 🎯 Kết quả mong đợi
- Hiểu rõ cách hoạt động của các cổng điều khiển
- Có thể tạo và phân tích các trạng thái rối lượng tử
- Biết cách thiết kế mạch lượng tử phức tạp

## 📖 Tài liệu tham khảo
- [Qiskit Multi-Qubit Gates](https://qiskit.org/textbook/ch-gates/multiple-qubits-entangled-states.html)
- [Entanglement in Qiskit](https://qiskit.org/textbook/ch-gates/entanglement.html) 