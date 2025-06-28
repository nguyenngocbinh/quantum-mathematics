# 📚 **Day 22: Quantum Phase Estimation**

---

## 🎯 **Mục tiêu học tập:**
- Hiểu sâu về quantum phase estimation
- Triển khai phase kickback technique
- Áp dụng eigenvalue estimation
- Thực hiện quantum counting algorithms

---

## 📖 **Lý thuyết cơ bản:**

### **1. Phase Kickback:**
Khi áp dụng controlled unitary operation $U$ lên eigenstate $|\psi\rangle$:

$$U|\psi\rangle = e^{2\pi i \phi}|\psi\rangle$$

Controlled operation tạo ra phase kickback:

$$|0\rangle|\psi\rangle \rightarrow |0\rangle|\psi\rangle$$
$$|1\rangle|\psi\rangle \rightarrow e^{2\pi i \phi}|1\rangle|\psi\rangle$$

### **2. Quantum Phase Estimation:**
Thuật toán để ước tính phase $\phi$ của eigenvalue $e^{2\pi i \phi}$:

1. **Preparation:** Tạo superposition trên precision qubits
2. **Controlled operations:** Áp dụng $U^{2^j}$ với controlled qubits
3. **Inverse QFT:** Chuyển đổi phase thành bit string
4. **Measurement:** Đọc kết quả phase

### **3. Độ chính xác:**
Với $t$ precision qubits, độ chính xác là $2^{-t}$.

---

## 💻 **Thực hành với Qiskit:**

### **Bài 1: Phase Estimation cơ bản**

```python
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

def controlled_phase_rotation(circuit, angle, control_qubit, target_qubit):
    """Controlled phase rotation"""
    circuit.cp(2 * np.pi * angle, control_qubit, target_qubit)

def phase_estimation_circuit(angle, precision_qubits=4):
    """Triển khai phase estimation hoàn chỉnh"""
    
    # Số qubit cho precision và eigenstate
    n_precision = precision_qubits
    n_eigenstate = 1
    total_qubits = n_precision + n_eigenstate
    
    qc = QuantumCircuit(total_qubits, n_precision)
    
    # Bước 1: Preparation - Hadamard trên precision qubits
    for i in range(n_precision):
        qc.h(i)
    
    # Bước 2: Controlled operations
    for i in range(n_precision):
        # Áp dụng U^(2^i) với controlled qubit i
        controlled_phase_rotation(qc, angle * (2**i), i, n_precision)
    
    # Bước 3: Inverse QFT
    # Hoán đổi qubits trước
    for i in range(n_precision//2):
        qc.swap(i, n_precision-1-i)
    
    # Inverse QFT rotations
    for i in range(n_precision):
        qc.h(i)
        for j in range(i+1, n_precision):
            qc.cp(-np.pi/float(2**(j-i)), i, j)
    
    # Bước 4: Measurement
    qc.measure(range(n_precision), range(n_precision))
    
    return qc

# Test với phase = 0.25 (1/4)
target_phase = 0.25
qc_pe = phase_estimation_circuit(target_phase, 4)

print("Phase Estimation Circuit:")
print(qc_pe)

# Chạy mạch
backend = Aer.get_backend('qasm_simulator')
job = execute(qc_pe, backend, shots=1000)
result = job.result()
counts = result.get_counts()

print(f"\nTarget phase: {target_phase}")
print(f"Results: {counts}")

# Chuyển đổi kết quả thành phase
for bitstring, count in counts.items():
    if count > 50:  # Chỉ xem kết quả có tần suất cao
        phase_estimate = int(bitstring, 2) / (2**4)
        print(f"Bitstring {bitstring} -> Phase estimate: {phase_estimate}")

plot_histogram(counts)
plt.show()
```

### **Bài 2: Eigenvalue Estimation**

```python
def eigenvalue_estimation(unitary_matrix, eigenvector, precision_qubits=4):
    """Ước tính eigenvalue của unitary matrix"""
    
    # Tạo mạch để encode eigenvector
    n_eigenstate = int(np.log2(len(eigenvector)))
    n_precision = precision_qubits
    total_qubits = n_precision + n_eigenstate
    
    qc = QuantumCircuit(total_qubits, n_precision)
    
    # Encode eigenvector (giả sử đơn giản)
    qc.x(n_precision)  # |1⟩ state
    
    # Hadamard trên precision qubits
    for i in range(n_precision):
        qc.h(i)
    
    # Controlled unitary operations
    # Trong thực tế, cần decompose unitary matrix
    for i in range(n_precision):
        # Giả sử U = phase rotation
        qc.cp(2 * np.pi * 0.3 * (2**i), i, n_precision)
    
    # Inverse QFT
    for i in range(n_precision//2):
        qc.swap(i, n_precision-1-i)
    
    for i in range(n_precision):
        qc.h(i)
        for j in range(i+1, n_precision):
            qc.cp(-np.pi/float(2**(j-i)), i, j)
    
    qc.measure(range(n_precision), range(n_precision))
    
    return qc

# Test eigenvalue estimation
eigenvector = [1, 0]  # |0⟩ state
unitary = np.array([[1, 0], [0, np.exp(2*np.pi*1j*0.3)]])
qc_ee = eigenvalue_estimation(unitary, eigenvector, 4)

backend = Aer.get_backend('qasm_simulator')
job = execute(qc_ee, backend, shots=1000)
result = job.result()
counts = result.get_counts()

print("Eigenvalue Estimation Results:")
print(counts)
```

### **Bài 3: Quantum Counting**

```python
def quantum_counting_oracle(marked_states, n_qubits):
    """Oracle cho quantum counting"""
    qc = QuantumCircuit(n_qubits + 1)  # +1 cho ancilla
    
    # Đánh dấu các trạng thái được chọn
    for state in marked_states:
        # Chuyển đổi state thành binary string
        binary = format(state, f'0{n_qubits}b')
        
        # Tạo controlled operation để flip ancilla
        controls = []
        for i, bit in enumerate(binary):
            if bit == '0':
                qc.x(i)
                controls.append(i)
            else:
                controls.append(i)
        
        # Multi-controlled NOT
        qc.mct(controls, n_qubits)
        
        # Reset controls
        for i, bit in enumerate(binary):
            if bit == '0':
                qc.x(i)
    
    return qc

def quantum_counting(marked_states, n_qubits, precision_qubits=6):
    """Quantum counting algorithm"""
    
    # Tạo oracle
    oracle = quantum_counting_oracle(marked_states, n_qubits)
    
    # Phase estimation circuit
    n_precision = precision_qubits
    total_qubits = n_precision + n_qubits + 1
    
    qc = QuantumCircuit(total_qubits, n_precision)
    
    # Hadamard trên precision qubits
    for i in range(n_precision):
        qc.h(i)
    
    # Controlled Grover iterations
    for i in range(n_precision):
        # Áp dụng Grover operator 2^i lần
        for _ in range(2**i):
            # Oracle
            qc.append(oracle, range(n_qubits + 1))
            # Diffusion
            qc.h(range(n_qubits))
            qc.x(range(n_qubits))
            qc.mct(range(n_qubits), n_qubits)
            qc.x(range(n_qubits))
            qc.h(range(n_qubits))
    
    # Inverse QFT
    for i in range(n_precision//2):
        qc.swap(i, n_precision-1-i)
    
    for i in range(n_precision):
        qc.h(i)
        for j in range(i+1, n_precision):
            qc.cp(-np.pi/float(2**(j-i)), i, j)
    
    qc.measure(range(n_precision), range(n_precision))
    
    return qc

# Test quantum counting
n_qubits = 3
marked_states = [1, 3, 5]  # 3 trạng thái được đánh dấu
qc_qc = quantum_counting(marked_states, n_qubits, 4)

backend = Aer.get_backend('qasm_simulator')
job = execute(qc_qc, backend, shots=1000)
result = job.result()
counts = result.get_counts()

print("Quantum Counting Results:")
print(counts)
print(f"Expected marked states: {len(marked_states)}")
```

---

## 🔬 **Bài tập thực hành:**

### **Bài tập 1: Phase Estimation với độ chính xác khác nhau**
So sánh độ chính xác của phase estimation với:
- 3 precision qubits
- 4 precision qubits  
- 5 precision qubits
- 6 precision qubits

### **Bài tập 2: Eigenvalue Estimation cho các matrix khác nhau**
Triển khai eigenvalue estimation cho:
- Pauli matrices (X, Y, Z)
- Hadamard matrix
- Rotation matrices

### **Bài tập 3: Quantum Counting với noise**
Thêm noise vào quantum counting và phân tích:
- Bit flip errors
- Phase errors
- Measurement errors

### **Bài tập 4: Phase Estimation với real hardware**
Chạy phase estimation trên:
- IBM Quantum Experience
- Qiskit Aer với noise models
- So sánh kết quả với simulator

---

## 📚 **Tài liệu tham khảo:**

### **Papers:**
- "Quantum Phase Estimation" - Kitaev (1995)
- "Quantum Counting" - Brassard et al. (1998)
- "Eigenvalue Estimation" - Abrams & Lloyd (1999)

### **Online Resources:**
- Qiskit Textbook: Quantum Phase Estimation
- IBM Quantum Experience: Phase Estimation Tutorial

---

## 🎯 **Kiểm tra kiến thức:**

1. **Câu hỏi:** Giải thích phase kickback mechanism?
2. **Câu hỏi:** Tại sao cần inverse QFT trong phase estimation?
3. **Câu hỏi:** Làm thế nào để cải thiện độ chính xác?
4. **Câu hỏi:** Ứng dụng của quantum counting?

---

## 🚀 **Chuẩn bị cho Day 23:**
- Ôn lại Bell states
- Hiểu quantum teleportation protocol
- Chuẩn bị cho classical communication

---

*"Quantum Phase Estimation is a fundamental algorithm that allows us to estimate the eigenvalues of unitary operators, which is crucial for many quantum algorithms."* - IBM Research 