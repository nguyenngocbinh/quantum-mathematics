# 📚 **Day 23: Quantum Teleportation Protocol**

---

## 🎯 **Mục tiêu học tập:**
- Hiểu nguyên lý quantum teleportation
- Triển khai Bell state preparation
- Thực hiện teleportation circuit
- Áp dụng classical communication protocols
- Hiểu error correction trong teleportation

---

## 📖 **Lý thuyết cơ bản:**

### **1. Bell States:**
Các trạng thái Bell là các trạng thái entangled cơ bản:

$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$
$$|\Phi^-\rangle = \frac{1}{\sqrt{2}}(|00\rangle - |11\rangle)$$
$$|\Psi^+\rangle = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)$$
$$|\Psi^-\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$$

### **2. Quantum Teleportation Protocol:**
Quá trình truyền thông tin lượng tử từ Alice đến Bob:

1. **Preparation:** Alice và Bob chia sẻ Bell state
2. **Entanglement:** Alice entangles qubit cần teleport với qubit của mình
3. **Measurement:** Alice đo Bell state và gửi kết quả cổ điển
4. **Correction:** Bob áp dụng correction dựa trên kết quả đo

### **3. Mathematical Description:**
Ban đầu: $|\psi\rangle_A \otimes |\Phi^+\rangle_{BC}$

Sau Bell measurement: $\frac{1}{2}\sum_{i,j} |\beta_{ij}\rangle_{AB} \otimes X^i Z^j |\psi\rangle_C$

---

## 💻 **Thực hành với Qiskit:**

### **Bài 1: Bell State Preparation**

```python
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram, plot_bloch_multivector
import matplotlib.pyplot as plt

def create_bell_state(bell_type='phi_plus'):
    """Tạo các Bell states khác nhau"""
    
    qc = QuantumCircuit(2, 2)
    
    # Bước 1: Hadamard trên qubit đầu tiên
    qc.h(0)
    
    # Bước 2: CNOT để tạo entanglement
    qc.cx(0, 1)
    
    # Bước 3: Áp dụng corrections cho các Bell states khác
    if bell_type == 'phi_minus':
        qc.z(0)
    elif bell_type == 'psi_plus':
        qc.x(1)
    elif bell_type == 'psi_minus':
        qc.x(1)
        qc.z(0)
    
    return qc

# Test tất cả Bell states
bell_states = ['phi_plus', 'phi_minus', 'psi_plus', 'psi_minus']

for bell_type in bell_states:
    qc = create_bell_state(bell_type)
    qc.measure_all()
    
    print(f"\nBell State: {bell_type}")
    print(qc)
    
    # Chạy mạch
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=1000)
    result = job.result()
    counts = result.get_counts()
    
    print(f"Results: {counts}")

# Visualize Bell state
qc_bell = create_bell_state('phi_plus')
backend_statevector = Aer.get_backend('statevector_simulator')
job = execute(qc_bell, backend_statevector)
result = job.result()
statevector = result.get_statevector()

print("\nBell State Vector:")
print(statevector)
```

### **Bài 2: Quantum Teleportation Circuit**

```python
def quantum_teleportation(initial_state='random'):
    """Triển khai quantum teleportation protocol"""
    
    # 3 qubits: Alice's qubit, Alice's Bell qubit, Bob's Bell qubit
    qc = QuantumCircuit(3, 2)
    
    # Bước 1: Chuẩn bị trạng thái cần teleport
    if initial_state == 'random':
        # Tạo trạng thái ngẫu nhiên
        qc.rx(np.pi/4, 0)  # Rotation X
        qc.ry(np.pi/3, 0)  # Rotation Y
    elif initial_state == '1':
        qc.x(0)
    elif initial_state == 'superposition':
        qc.h(0)
    
    # Bước 2: Tạo Bell state giữa qubit 1 và 2
    qc.h(1)
    qc.cx(1, 2)
    
    # Bước 3: Bell measurement (Alice đo qubit 0 và 1)
    qc.cx(0, 1)
    qc.h(0)
    
    # Bước 4: Measurement
    qc.measure([0, 1], [0, 1])
    
    # Bước 5: Classical communication và correction
    # (Trong thực tế, Bob sẽ nhận thông tin cổ điển và áp dụng correction)
    
    return qc

def teleportation_with_correction(initial_state='random'):
    """Teleportation với correction được áp dụng"""
    
    qc = QuantumCircuit(3, 2)
    
    # Chuẩn bị trạng thái
    if initial_state == 'random':
        qc.rx(np.pi/4, 0)
        qc.ry(np.pi/3, 0)
    elif initial_state == '1':
        qc.x(0)
    elif initial_state == 'superposition':
        qc.h(0)
    
    # Tạo Bell state
    qc.h(1)
    qc.cx(1, 2)
    
    # Bell measurement
    qc.cx(0, 1)
    qc.h(0)
    
    # Measurement
    qc.measure([0, 1], [0, 1])
    
    # Correction (dựa trên kết quả đo)
    # Trong thực tế, Bob sẽ áp dụng X và Z gates dựa trên classical bits
    qc.x(2).c_if(0, 1)  # Apply X if bit 0 is 1
    qc.z(2).c_if(1, 1)  # Apply Z if bit 1 is 1
    
    return qc

# Test teleportation
initial_states = ['0', '1', 'superposition', 'random']

for state in initial_states:
    qc = quantum_teleportation(state)
    
    print(f"\nTeleportation: {state}")
    print(qc)
    
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=1000)
    result = job.result()
    counts = result.get_counts()
    
    print(f"Measurement results: {counts}")
```

### **Bài 3: Teleportation với Error Correction**

```python
def teleportation_with_noise(error_rate=0.1):
    """Teleportation với noise và error correction"""
    
    from qiskit.providers.aer.noise import NoiseModel
    from qiskit.providers.aer.noise.errors import depolarizing_error
    
    # Tạo noise model
    noise_model = NoiseModel()
    error = depolarizing_error(error_rate, 1)
    noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])
    
    # Teleportation circuit
    qc = teleportation_with_correction('superposition')
    
    # Chạy với noise
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=1000, noise_model=noise_model)
    result = job.result()
    counts_with_noise = result.get_counts()
    
    # Chạy không có noise để so sánh
    job_perfect = execute(qc, backend, shots=1000)
    result_perfect = job_perfect.result()
    counts_perfect = result_perfect.get_counts()
    
    print(f"Error rate: {error_rate}")
    print(f"With noise: {counts_with_noise}")
    print(f"Perfect: {counts_perfect}")
    
    return counts_with_noise, counts_perfect

# Test với các error rates khác nhau
error_rates = [0.01, 0.05, 0.1, 0.2]

for rate in error_rates:
    print(f"\nTesting with error rate: {rate}")
    noisy, perfect = teleportation_with_noise(rate)
```

### **Bài 4: Multi-qubit Teleportation**

```python
def multi_qubit_teleportation(n_qubits=2):
    """Teleportation cho nhiều qubit"""
    
    # Số qubit: n qubit cần teleport + n Bell pairs
    total_qubits = 3 * n_qubits
    qc = QuantumCircuit(total_qubits, 2 * n_qubits)
    
    # Chuẩn bị trạng thái cần teleport
    for i in range(n_qubits):
        qc.h(i)  # Superposition state
    
    # Tạo Bell states
    for i in range(n_qubits):
        qc.h(n_qubits + 2*i)
        qc.cx(n_qubits + 2*i, n_qubits + 2*i + 1)
    
    # Bell measurements
    for i in range(n_qubits):
        qc.cx(i, n_qubits + 2*i)
        qc.h(i)
    
    # Measurements
    measurement_qubits = list(range(n_qubits)) + list(range(n_qubits, 2*n_qubits))
    qc.measure(measurement_qubits, range(2*n_qubits))
    
    return qc

# Test multi-qubit teleportation
qc_multi = multi_qubit_teleportation(2)
print("Multi-qubit Teleportation:")
print(qc_multi)

backend = Aer.get_backend('qasm_simulator')
job = execute(qc_multi, backend, shots=1000)
result = job.result()
counts = result.get_counts()

print(f"Results: {counts}")
```

---

## 🔬 **Bài tập thực hành:**

### **Bài tập 1: Bell State Discrimination**
Tạo circuit để phân biệt các Bell states:
- Input: Bell state chưa biết
- Output: Classical bits xác định Bell state

### **Bài tập 2: Teleportation với Different Gates**
Teleport qubit qua các gates khác nhau:
- Hadamard gate
- Phase gate
- Rotation gates

### **Bài tập 3: Teleportation Network**
Tạo network teleportation:
- Alice → Bob → Charlie
- Multiple Bell states
- Error propagation analysis

### **Bài tập 4: Teleportation với Real Hardware**
Chạy teleportation trên:
- IBM Quantum Experience
- So sánh với simulator
- Analyze error rates

---

## 📚 **Tài liệu tham khảo:**

### **Papers:**
- "Teleporting an Unknown Quantum State" - Bennett et al. (1993)
- "Quantum Teleportation" - Bouwmeester et al. (1997)
- "Bell States" - Bell (1964)

### **Online Resources:**
- Qiskit Textbook: Quantum Teleportation
- IBM Quantum Experience: Teleportation Tutorial

---

## 🎯 **Kiểm tra kiến thức:**

1. **Câu hỏi:** Tại sao cần classical communication trong teleportation?
2. **Câu hỏi:** Làm thế nào để tạo Bell states?
3. **Câu hỏi:** Các loại errors trong teleportation?
4. **Câu hỏi:** Ứng dụng của quantum teleportation?

---

## 🚀 **Chuẩn bị cho Day 24:**
- Ôn lại error correction basics
- Hiểu syndrome measurement
- Chuẩn bị cho quantum error correction codes

---

*"Quantum teleportation allows us to transfer quantum information from one location to another using entanglement and classical communication."* - IBM Research 