# 📚 **Day 24: Quantum Error Correction Codes**

---

## 🎯 **Mục tiêu học tập:**
- Hiểu nguyên lý quantum error correction
- Triển khai 3-qubit code và 5-qubit code
- Thực hiện syndrome measurement
- Áp dụng error detection và correction
- Hiểu surface codes và topological codes

---

## 📖 **Lý thuyết cơ bản:**

### **1. Types of Quantum Errors:**
- **Bit flip errors (X):** $|0\rangle \leftrightarrow |1\rangle$
- **Phase flip errors (Z):** $|0\rangle \rightarrow |0\rangle, |1\rangle \rightarrow -|1\rangle$
- **Combined errors (Y):** $Y = iXZ$

### **2. 3-Qubit Code:**
Bảo vệ chống bit flip errors:

**Encoding:**
$|0\rangle \rightarrow |000\rangle$
$|1\rangle \rightarrow |111\rangle$

**Syndrome measurement:**
$M_1 = Z_1 Z_2$
$M_2 = Z_2 Z_3$

### **3. 5-Qubit Code:**
Bảo vệ chống cả bit flip và phase flip:

**Stabilizers:**
$S_1 = X_1 Z_2 Z_3 X_4$
$S_2 = X_2 Z_3 Z_4 X_5$
$S_3 = X_1 X_3 Z_4 Z_5$
$S_4 = Z_1 X_2 X_4 Z_5$

---

## 💻 **Thực hành với Qiskit:**

### **Bài 1: 3-Qubit Code Implementation**

```python
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

def three_qubit_encoding():
    """3-qubit code encoding"""
    qc = QuantumCircuit(3, 2)  # 3 data qubits, 2 syndrome qubits
    
    # Encoding: |ψ⟩ → |ψ⟩|00⟩
    # CNOT từ qubit 0 đến qubit 1 và 2
    qc.cx(0, 1)
    qc.cx(0, 2)
    
    return qc

def three_qubit_syndrome_measurement():
    """Syndrome measurement cho 3-qubit code"""
    qc = QuantumCircuit(5, 2)  # 3 data + 2 syndrome qubits
    
    # Syndrome qubits bắt đầu ở |0⟩
    # M1 = Z1 Z2
    qc.cx(0, 3)  # CNOT từ qubit 0 đến syndrome qubit 0
    qc.cx(1, 3)  # CNOT từ qubit 1 đến syndrome qubit 0
    
    # M2 = Z2 Z3
    qc.cx(1, 4)  # CNOT từ qubit 1 đến syndrome qubit 1
    qc.cx(2, 4)  # CNOT từ qubit 2 đến syndrome qubit 1
    
    # Measure syndrome qubits
    qc.measure([3, 4], [0, 1])
    
    return qc

def three_qubit_error_correction():
    """3-qubit code với error correction"""
    qc = QuantumCircuit(5, 2)
    
    # Encoding
    qc.cx(0, 1)
    qc.cx(0, 2)
    
    # Introduce error (bit flip on qubit 1)
    qc.x(1)
    
    # Syndrome measurement
    qc.cx(0, 3)
    qc.cx(1, 3)
    qc.cx(1, 4)
    qc.cx(2, 4)
    
    # Measure syndrome
    qc.measure([3, 4], [0, 1])
    
    return qc

# Test 3-qubit code
print("3-Qubit Code Test:")
qc_3bit = three_qubit_error_correction()
print(qc_3bit)

backend = Aer.get_backend('qasm_simulator')
job = execute(qc_3bit, backend, shots=1000)
result = job.result()
counts = result.get_counts()

print(f"Syndrome measurement results: {counts}")
# Syndrome '01' indicates error on qubit 1
# Syndrome '10' indicates error on qubit 0
# Syndrome '11' indicates error on qubit 2
```

### **Bài 2: 5-Qubit Code Implementation**

```python
def five_qubit_encoding():
    """5-qubit code encoding"""
    qc = QuantumCircuit(5, 4)  # 5 data qubits, 4 syndrome qubits
    
    # 5-qubit code encoding circuit
    # This is a simplified version
    qc.h(1)
    qc.h(2)
    qc.h(3)
    qc.h(4)
    
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.cx(3, 4)
    
    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.h(3)
    qc.h(4)
    
    return qc

def five_qubit_stabilizers():
    """5-qubit code stabilizer measurements"""
    qc = QuantumCircuit(9, 4)  # 5 data + 4 syndrome qubits
    
    # Stabilizer S1 = X1 Z2 Z3 X4
    qc.h(5)  # Syndrome qubit 0
    qc.cx(5, 0)  # X1
    qc.cz(5, 1)  # Z2
    qc.cz(5, 2)  # Z3
    qc.cx(5, 3)  # X4
    qc.h(5)
    
    # Stabilizer S2 = X2 Z3 Z4 X5
    qc.h(6)  # Syndrome qubit 1
    qc.cx(6, 1)  # X2
    qc.cz(6, 2)  # Z3
    qc.cz(6, 3)  # Z4
    qc.cx(6, 4)  # X5
    qc.h(6)
    
    # Stabilizer S3 = X1 X3 Z4 Z5
    qc.h(7)  # Syndrome qubit 2
    qc.cx(7, 0)  # X1
    qc.cx(7, 2)  # X3
    qc.cz(7, 3)  # Z4
    qc.cz(7, 4)  # Z5
    qc.h(7)
    
    # Stabilizer S4 = Z1 X2 X4 Z5
    qc.h(8)  # Syndrome qubit 3
    qc.cz(8, 0)  # Z1
    qc.cx(8, 1)  # X2
    qc.cx(8, 3)  # X4
    qc.cz(8, 4)  # Z5
    qc.h(8)
    
    # Measure syndrome qubits
    qc.measure([5, 6, 7, 8], [0, 1, 2, 3])
    
    return qc

def five_qubit_error_correction():
    """5-qubit code với error correction"""
    qc = QuantumCircuit(9, 4)
    
    # Encoding
    qc.h(1)
    qc.h(2)
    qc.h(3)
    qc.h(4)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.cx(3, 4)
    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.h(3)
    qc.h(4)
    
    # Introduce error (bit flip on qubit 2)
    qc.x(2)
    
    # Stabilizer measurements
    # S1
    qc.h(5)
    qc.cx(5, 0)
    qc.cz(5, 1)
    qc.cz(5, 2)
    qc.cx(5, 3)
    qc.h(5)
    
    # S2
    qc.h(6)
    qc.cx(6, 1)
    qc.cz(6, 2)
    qc.cz(6, 3)
    qc.cx(6, 4)
    qc.h(6)
    
    # S3
    qc.h(7)
    qc.cx(7, 0)
    qc.cx(7, 2)
    qc.cz(7, 3)
    qc.cz(7, 4)
    qc.h(7)
    
    # S4
    qc.h(8)
    qc.cz(8, 0)
    qc.cx(8, 1)
    qc.cx(8, 3)
    qc.cz(8, 4)
    qc.h(8)
    
    # Measure syndrome
    qc.measure([5, 6, 7, 8], [0, 1, 2, 3])
    
    return qc

# Test 5-qubit code
print("\n5-Qubit Code Test:")
qc_5bit = five_qubit_error_correction()
print(qc_5bit)

backend = Aer.get_backend('qasm_simulator')
job = execute(qc_5bit, backend, shots=1000)
result = job.result()
counts = result.get_counts()

print(f"Syndrome measurement results: {counts}")
```

### **Bài 3: Error Detection và Correction**

```python
def error_detection_circuit(error_type='bit_flip', error_location=1):
    """Circuit để detect và correct errors"""
    
    if error_type == 'bit_flip':
        qc = QuantumCircuit(5, 2)
        
        # Encoding
        qc.cx(0, 1)
        qc.cx(0, 2)
        
        # Introduce bit flip error
        if error_location == 0:
            qc.x(0)
        elif error_location == 1:
            qc.x(1)
        elif error_location == 2:
            qc.x(2)
        
        # Syndrome measurement
        qc.cx(0, 3)
        qc.cx(1, 3)
        qc.cx(1, 4)
        qc.cx(2, 4)
        
        # Measure syndrome
        qc.measure([3, 4], [0, 1])
        
        return qc
    
    elif error_type == 'phase_flip':
        qc = QuantumCircuit(5, 2)
        
        # Encoding for phase flip protection
        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.cx(0, 1)
        qc.cx(1, 2)
        
        # Introduce phase flip error
        if error_location == 0:
            qc.z(0)
        elif error_location == 1:
            qc.z(1)
        elif error_location == 2:
            qc.z(2)
        
        # Syndrome measurement for phase flip
        qc.cx(0, 3)
        qc.cx(1, 3)
        qc.cx(1, 4)
        qc.cx(2, 4)
        
        # Measure syndrome
        qc.measure([3, 4], [0, 1])
        
        return qc

def error_correction_lookup_table():
    """Lookup table cho error correction"""
    
    # 3-qubit code syndrome table
    syndrome_table = {
        '00': 'No error',
        '01': 'Bit flip on qubit 1',
        '10': 'Bit flip on qubit 0', 
        '11': 'Bit flip on qubit 2'
    }
    
    return syndrome_table

# Test error detection
error_types = ['bit_flip', 'phase_flip']
error_locations = [0, 1, 2]

for error_type in error_types:
    print(f"\nTesting {error_type} errors:")
    for location in error_locations:
        qc = error_detection_circuit(error_type, location)
        
        backend = Aer.get_backend('qasm_simulator')
        job = execute(qc, backend, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        # Find most common syndrome
        most_common = max(counts, key=counts.get)
        print(f"Error at qubit {location}: Syndrome {most_common}")

# Error correction lookup
syndrome_table = error_correction_lookup_table()
print(f"\nError Correction Lookup Table:")
for syndrome, error in syndrome_table.items():
    print(f"Syndrome {syndrome}: {error}")
```

### **Bài 4: Surface Code Basics**

```python
def surface_code_plaquette():
    """Surface code plaquette operator"""
    qc = QuantumCircuit(4, 1)
    
    # Plaquette operator: Z1 Z2 Z3 Z4
    qc.cz(0, 1)
    qc.cz(1, 2)
    qc.cz(2, 3)
    qc.cz(3, 0)
    
    # Measure ancilla qubit
    qc.measure(0, 0)
    
    return qc

def surface_code_star():
    """Surface code star operator"""
    qc = QuantumCircuit(4, 1)
    
    # Star operator: X1 X2 X3 X4
    qc.h(0)  # Hadamard để measure X
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(0, 3)
    qc.h(0)
    
    # Measure ancilla qubit
    qc.measure(0, 0)
    
    return qc

def surface_code_logical_qubit():
    """Surface code logical qubit encoding"""
    # Simplified surface code
    qc = QuantumCircuit(9, 4)  # 9 data qubits, 4 syndrome qubits
    
    # Encode logical qubit
    # This is a simplified version of surface code encoding
    
    # Apply stabilizers
    # Plaquette operators
    qc.cz(0, 1)
    qc.cz(1, 2)
    qc.cz(2, 3)
    qc.cz(3, 0)
    
    qc.cz(4, 5)
    qc.cz(5, 6)
    qc.cz(6, 7)
    qc.cz(7, 4)
    
    # Star operators
    qc.h(8)
    qc.cx(8, 0)
    qc.cx(8, 1)
    qc.cx(8, 4)
    qc.cx(8, 5)
    qc.h(8)
    
    # Measure syndrome qubits
    qc.measure([0, 1, 4, 5], [0, 1, 2, 3])
    
    return qc

# Test surface code components
print("\nSurface Code Components:")

# Plaquette operator
qc_plaquette = surface_code_plaquette()
print("Plaquette Operator:")
print(qc_plaquette)

# Star operator
qc_star = surface_code_star()
print("\nStar Operator:")
print(qc_star)

# Logical qubit
qc_logical = surface_code_logical_qubit()
print("\nLogical Qubit Encoding:")
print(qc_logical)

backend = Aer.get_backend('qasm_simulator')
job = execute(qc_logical, backend, shots=1000)
result = job.result()
counts = result.get_counts()

print(f"Surface code syndrome results: {counts}")
```

---

## 🔬 **Bài tập thực hành:**

### **Bài tập 1: Error Rate Analysis**
Phân tích error rates cho các codes khác nhau:
- 3-qubit code vs 5-qubit code
- Bit flip vs phase flip errors
- Multiple error scenarios

### **Bài tập 2: Custom Error Correction Codes**
Thiết kế custom error correction codes:
- 4-qubit code
- 6-qubit code
- Analyze error correction capabilities

### **Bài tập 3: Fault-Tolerant Gates**
Triển khai fault-tolerant gates:
- Fault-tolerant CNOT
- Fault-tolerant Hadamard
- Error propagation analysis

### **Bài tập 4: Surface Code Implementation**
Triển khai surface code hoàn chỉnh:
- Logical qubit encoding
- Error detection
- Error correction
- Code distance analysis

---

## 📚 **Tài liệu tham khảo:**

### **Papers:**
- "Quantum Error Correction" - Shor (1995)
- "5-Qubit Code" - Laflamme et al. (1996)
- "Surface Codes" - Kitaev (2003)

### **Online Resources:**
- Qiskit Textbook: Quantum Error Correction
- IBM Quantum Experience: Error Correction Tutorial

---

## 🎯 **Kiểm tra kiến thức:**

1. **Câu hỏi:** Tại sao cần quantum error correction?
2. **Câu hỏi:** Sự khác biệt giữa 3-qubit và 5-qubit codes?
3. **Câu hỏi:** Làm thế nào để detect và correct errors?
4. **Câu hỏi:** Ưu điểm của surface codes?

---

## 🚀 **Chuẩn bị cho Day 25:**
- Ôn lại search algorithms
- Hiểu oracle design
- Chuẩn bị cho Grover's algorithm

---

*"Quantum error correction is essential for building reliable quantum computers that can perform complex computations despite the presence of noise and errors."* - IBM Research 