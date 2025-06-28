# 📚 **Day 21: Quantum Fourier Transform (QFT)**

---

## 🎯 **Mục tiêu học tập:**
- Hiểu nguyên lý hoạt động của Quantum Fourier Transform
- Triển khai QFT bằng Qiskit
- Áp dụng QFT trong phase estimation và period finding
- Hiểu ứng dụng của QFT trong các thuật toán lượng tử

---

## 📖 **Lý thuyết cơ bản:**

### **1. Fourier Transform cổ điển:**
Fourier Transform chuyển đổi tín hiệu từ miền thời gian sang miền tần số:

$$F(\omega) = \int_{-\infty}^{\infty} f(t) e^{-i\omega t} dt$$

### **2. Quantum Fourier Transform:**
QFT chuyển đổi trạng thái lượng tử từ computational basis sang Fourier basis:

$$QFT|j\rangle = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} e^{2\pi i jk/N} |k\rangle$$

Trong đó $N = 2^n$ với $n$ là số qubit.

### **3. Cấu trúc mạch QFT:**
QFT có thể được triển khai bằng các cổng Hadamard và controlled-phase:

```
H ---•---•---•---•---
      |   |   |   |
     R₂  R₃  R₄  R₅
      |   |   |   |
     H ---•---•---•---
          |   |   |
         R₂  R₃  R₄
          |   |   |
         H ---•---•---
              |   |
             R₂  R₃
              |   |
             H ---•---
                  |
                 R₂
                  |
                 H
```

---

## 💻 **Thực hành với Qiskit:**

### **Bài 1: Triển khai QFT cơ bản**

```python
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

def qft_rotations(circuit, n):
    """Thực hiện QFT rotations trên n qubit"""
    if n == 0:
        return circuit
    
    n -= 1
    circuit.h(n)
    
    for qubit in range(n):
        circuit.cp(np.pi/float(2**(n-qubit)), qubit, n)
    
    qft_rotations(circuit, n)

def swap_registers(circuit, n):
    """Hoán đổi các qubit để có thứ tự đúng"""
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit

def qft(circuit, n):
    """Triển khai QFT hoàn chỉnh"""
    qft_rotations(circuit, n)
    swap_registers(circuit, n)
    return circuit

# Tạo mạch QFT 3-qubit
n_qubits = 3
qc = QuantumCircuit(n_qubits)
qc.x(0)  # Đặt input |001⟩
qft(qc, n_qubits)
qc.measure_all()

# Chạy mạch
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1000)
result = job.result()
counts = result.get_counts()

print("QFT Circuit:")
print(qc)
print("\nResults:")
print(counts)
plot_histogram(counts)
plt.show()
```

### **Bài 2: Phase Estimation với QFT**

```python
def phase_estimation(angle, precision_qubits=3):
    """Phase estimation để ước tính phase của một góc"""
    
    # Số qubit cho precision
    n_precision = precision_qubits
    # Qubit cho eigenstate
    n_eigenstate = 1
    
    qc = QuantumCircuit(n_precision + n_eigenstate, n_precision)
    
    # Áp dụng Hadamard trên precision qubits
    for qubit in range(n_precision):
        qc.h(qubit)
    
    # Controlled phase rotations
    for i in range(n_precision):
        qc.cp(2 * np.pi * angle * (2**i), n_precision, i)
    
    # Inverse QFT
    qc.h(0)
    for i in range(1, n_precision):
        qc.cp(-np.pi/float(2**(i)), 0, i)
    qc.h(1)
    for i in range(2, n_precision):
        qc.cp(-np.pi/float(2**(i-1)), 1, i)
    # ... tiếp tục cho các qubit khác
    
    qc.measure(range(n_precision), range(n_precision))
    
    return qc

# Test với phase = 0.125 (1/8)
angle = 0.125
qc_pe = phase_estimation(angle, 4)

backend = Aer.get_backend('qasm_simulator')
job = execute(qc_pe, backend, shots=1000)
result = job.result()
counts = result.get_counts()

print("Phase Estimation Circuit:")
print(qc_pe)
print(f"\nExpected phase: {angle}")
print(f"Results: {counts}")
```

### **Bài 3: Period Finding với QFT**

```python
def period_finding(a, N, n_qubits=8):
    """Tìm chu kỳ của hàm f(x) = a^x mod N"""
    
    qc = QuantumCircuit(2*n_qubits, n_qubits)
    
    # Superposition trên register đầu tiên
    for i in range(n_qubits):
        qc.h(i)
    
    # Oracle để tính f(x) = a^x mod N
    # (Đây là phiên bản đơn giản, trong thực tế cần modular exponentiation)
    for i in range(n_qubits):
        qc.cx(i, n_qubits + i)
    
    # QFT trên register đầu tiên
    qc.h(0)
    for i in range(1, n_qubits):
        qc.cp(np.pi/float(2**(i)), 0, i)
    qc.h(1)
    for i in range(2, n_qubits):
        qc.cp(np.pi/float(2**(i-1)), 1, i)
    # ... tiếp tục
    
    qc.measure(range(n_qubits), range(n_qubits))
    
    return qc

# Test period finding
a, N = 2, 15
qc_pf = period_finding(a, N, 4)

backend = Aer.get_backend('qasm_simulator')
job = execute(qc_pf, backend, shots=1000)
result = job.result()
counts = result.get_counts()

print("Period Finding Circuit:")
print(qc_pf)
print(f"\nLooking for period of {a}^x mod {N}")
print(f"Results: {counts}")
```

---

## 🔬 **Bài tập thực hành:**

### **Bài tập 1: QFT với input khác nhau**
Tạo mạch QFT và test với các input khác nhau:
- |000⟩ → |000⟩ + |001⟩ + ... + |111⟩
- |100⟩ → Superposition với phase khác nhau
- |111⟩ → Superposition với phase khác nhau

### **Bài tập 2: Inverse QFT**
Triển khai inverse QFT và verify rằng QFT⁻¹(QFT|ψ⟩) = |ψ⟩

### **Bài tập 3: Phase Estimation chính xác**
Cải thiện độ chính xác của phase estimation bằng cách:
- Tăng số qubit precision
- Sử dụng error mitigation
- So sánh với giá trị lý thuyết

### **Bài tập 4: Period Finding thực tế**
Triển khai period finding cho các trường hợp:
- a = 2, N = 15 (period = 4)
- a = 3, N = 15 (period = 4)
- a = 7, N = 15 (period = 4)

---

## 📚 **Tài liệu tham khảo:**

### **Sách:**
- "Quantum Computation and Quantum Information" - Nielsen & Chuang (Chapter 5)
- "Programming Quantum Computers" - Johnston et al. (Chapter 8)

### **Papers:**
- "Quantum Fourier Transform" - Coppersmith (1994)
- "Quantum Phase Estimation" - Kitaev (1995)

### **Online Resources:**
- Qiskit Textbook: Quantum Fourier Transform
- IBM Quantum Experience: QFT Tutorial

---

## 🎯 **Kiểm tra kiến thức:**

1. **Câu hỏi:** Tại sao QFT quan trọng trong computing lượng tử?
2. **Câu hỏi:** Làm thế nào để triển khai QFT với n qubit?
3. **Câu hỏi:** Mối quan hệ giữa QFT và phase estimation?
4. **Câu hỏi:** Ứng dụng của QFT trong period finding?

---

## 🚀 **Chuẩn bị cho Day 22:**
- Ôn lại phase estimation
- Hiểu eigenvalue estimation
- Chuẩn bị cho quantum counting

---

*"The Quantum Fourier Transform is the quantum analogue of the discrete Fourier transform, and it's a key component in many quantum algorithms."* - Qiskit Textbook 