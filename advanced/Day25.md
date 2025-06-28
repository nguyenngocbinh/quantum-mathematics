# Day 25: Thuật Toán Grover - Tìm Kiếm Lượng Tử

## 🎯 Mục tiêu
- Hiểu thuật toán Grover và nguyên lý hoạt động
- Triển khai thuật toán Grover trong Qiskit
- Phân tích hiệu suất và ứng dụng thực tế

## 🔍 Thuật Toán Grover - Tổng Quan

### Nguyên lý cơ bản
Thuật toán Grover là thuật toán lượng tử để tìm kiếm trong cơ sở dữ liệu không có cấu trúc với độ phức tạp O(√N) thay vì O(N) của thuật toán cổ điển.

```python
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt

def grover_oracle(marked_state):
    """
    Tạo oracle cho thuật toán Grover
    marked_state: trạng thái cần tìm (ví dụ: '11' cho 2 qubit)
    """
    n_qubits = len(marked_state)
    qc = QuantumCircuit(n_qubits)
    
    # Đánh dấu trạng thái cần tìm bằng phase kickback
    for i, bit in enumerate(marked_state):
        if bit == '0':
            qc.x(i)
    
    # Áp dụng multi-controlled Z gate
    if n_qubits == 1:
        qc.z(0)
    elif n_qubits == 2:
        qc.cz(0, 1)
    elif n_qubits == 3:
        qc.ccz(0, 1, 2)
    else:
        # Sử dụng Toffoli gates cho nhiều qubit hơn
        qc.mct(list(range(n_qubits-1)), n_qubits-1)
    
    # Đảo ngược các X gates
    for i, bit in enumerate(marked_state):
        if bit == '0':
            qc.x(i)
    
    return qc

def grover_diffusion(n_qubits):
    """
    Tạo diffusion operator (Grover diffusion)
    """
    qc = QuantumCircuit(n_qubits)
    
    # Áp dụng Hadamard trên tất cả qubit
    for i in range(n_qubits):
        qc.h(i)
    
    # Áp dụng X trên tất cả qubit
    for i in range(n_qubits):
        qc.x(i)
    
    # Áp dụng multi-controlled Z
    if n_qubits == 1:
        qc.z(0)
    elif n_qubits == 2:
        qc.cz(0, 1)
    elif n_qubits == 3:
        qc.ccz(0, 1, 2)
    else:
        qc.mct(list(range(n_qubits-1)), n_qubits-1)
    
    # Đảo ngược X gates
    for i in range(n_qubits):
        qc.x(i)
    
    # Đảo ngược Hadamard gates
    for i in range(n_qubits):
        qc.h(i)
    
    return qc
```

## 🔧 Triển Khai Thuật Toán Grover

### 1. Grover cho 2 qubit (tìm kiếm trong 4 trạng thái)

```python
def grover_2_qubit(marked_state='11', iterations=1):
    """
    Thuật toán Grover cho 2 qubit
    marked_state: trạng thái cần tìm ('00', '01', '10', '11')
    iterations: số lần lặp Grover
    """
    qc = QuantumCircuit(2, 2)
    
    # Bước 1: Khởi tạo siêu vị đều
    qc.h(0)
    qc.h(1)
    
    # Bước 2: Lặp Grover
    for _ in range(iterations):
        # Oracle
        oracle = grover_oracle(marked_state)
        qc = qc.compose(oracle)
        
        # Diffusion
        diffusion = grover_diffusion(2)
        qc = qc.compose(diffusion)
    
    # Bước 3: Đo lường
    qc.measure([0, 1], [0, 1])
    
    return qc

# Thử nghiệm với các trạng thái khác nhau
def test_grover_2_qubit():
    marked_states = ['00', '01', '10', '11']
    results = {}
    
    backend = Aer.get_backend('qasm_simulator')
    
    for state in marked_states:
        qc = grover_2_qubit(state, iterations=1)
        job = execute(qc, backend, shots=1000)
        result = job.result()
        results[state] = result.get_counts(qc)
        print(f"Tìm kiếm {state}: {results[state]}")
    
    return results
```

### 2. Grover cho 3 qubit (tìm kiếm trong 8 trạng thái)

```python
def grover_3_qubit(marked_state='111', iterations=2):
    """
    Thuật toán Grover cho 3 qubit
    marked_state: trạng thái cần tìm
    iterations: số lần lặp (tối ưu cho 3 qubit là 2 lần)
    """
    qc = QuantumCircuit(3, 3)
    
    # Khởi tạo siêu vị đều
    for i in range(3):
        qc.h(i)
    
    # Lặp Grover
    for _ in range(iterations):
        # Oracle
        oracle = grover_oracle(marked_state)
        qc = qc.compose(oracle)
        
        # Diffusion
        diffusion = grover_diffusion(3)
        qc = qc.compose(diffusion)
    
    # Đo lường
    qc.measure([0, 1, 2], [0, 1, 2])
    
    return qc
```

## 📊 Phân Tích Hiệu Suất

### 1. Tính toán số lần lặp tối ưu

```python
def optimal_grover_iterations(n_qubits):
    """
    Tính số lần lặp tối ưu cho thuật toán Grover
    """
    N = 2**n_qubits
    optimal = int(np.pi/4 * np.sqrt(N))
    return optimal

def grover_success_probability(n_qubits, iterations):
    """
    Tính xác suất thành công của thuật toán Grover
    """
    N = 2**n_qubits
    theta = np.arcsin(1/np.sqrt(N))
    angle = (2*iterations + 1) * theta
    return np.sin(angle)**2

# Phân tích cho các số qubit khác nhau
def analyze_grover_performance():
    qubit_counts = [2, 3, 4, 5, 6]
    analysis = {}
    
    for n in qubit_counts:
        optimal_iters = optimal_grover_iterations(n)
        success_prob = grover_success_probability(n, optimal_iters)
        
        analysis[n] = {
            'database_size': 2**n,
            'optimal_iterations': optimal_iters,
            'success_probability': success_prob,
            'classical_complexity': 2**n,
            'quantum_complexity': optimal_iters
        }
    
    return analysis
```

### 2. So sánh với tìm kiếm cổ điển

```python
def classical_search_simulation(database_size, marked_element):
    """
    Mô phỏng tìm kiếm cổ điển
    """
    import random
    
    # Tạo database ngẫu nhiên
    database = list(range(database_size))
    random.shuffle(database)
    
    # Tìm kiếm tuần tự
    comparisons = 0
    for i, element in enumerate(database):
        comparisons += 1
        if element == marked_element:
            return comparisons
    
    return comparisons

def compare_search_methods():
    """
    So sánh hiệu suất tìm kiếm cổ điển vs lượng tử
    """
    sizes = [4, 8, 16, 32, 64]
    results = {}
    
    for size in sizes:
        # Mô phỏng cổ điển
        classical_avg = np.mean([
            classical_search_simulation(size, 0) 
            for _ in range(100)
        ])
        
        # Tính toán lượng tử
        n_qubits = int(np.log2(size))
        quantum_iters = optimal_grover_iterations(n_qubits)
        
        results[size] = {
            'classical_average': classical_avg,
            'quantum_iterations': quantum_iters,
            'speedup': classical_avg / quantum_iters
        }
    
    return results
```

## 🎯 Ứng Dụng Thực Tế

### 1. Tìm kiếm trong cơ sở dữ liệu

```python
def database_search_example():
    """
    Ví dụ tìm kiếm trong database đơn giản
    """
    # Database: ['Alice', 'Bob', 'Charlie', 'David']
    # Tìm kiếm 'Charlie' (index 2)
    
    qc = QuantumCircuit(2, 2)
    
    # Khởi tạo
    qc.h(0)
    qc.h(1)
    
    # Oracle cho 'Charlie' (binary: 10)
    qc.x(1)  # Đảo bit thứ 2
    qc.cz(0, 1)
    qc.x(1)
    
    # Diffusion
    qc.h(0)
    qc.h(1)
    qc.x(0)
    qc.x(1)
    qc.cz(0, 1)
    qc.x(0)
    qc.x(1)
    qc.h(0)
    qc.h(1)
    
    qc.measure([0, 1], [0, 1])
    
    return qc
```

### 2. Giải bài toán SAT

```python
def sat_oracle(clause):
    """
    Tạo oracle cho bài toán SAT
    clause: tuple của các literal (ví dụ: (1, -2, 3) cho x1 OR NOT x2 OR x3)
    """
    n_vars = max(abs(x) for x in clause)
    qc = QuantumCircuit(n_vars)
    
    # Áp dụng logic của clause
    for literal in clause:
        if literal < 0:
            qc.x(abs(literal) - 1)
    
    # Multi-controlled Z
    if n_vars == 1:
        qc.z(0)
    elif n_vars == 2:
        qc.cz(0, 1)
    else:
        qc.mct(list(range(n_vars-1)), n_vars-1)
    
    # Đảo ngược
    for literal in clause:
        if literal < 0:
            qc.x(abs(literal) - 1)
    
    return qc
```

## 📚 Bài Tập Thực Hành

### Bài tập 1: Tối ưu hóa số lần lặp
```python
def find_optimal_iterations(n_qubits):
    """
    Tìm số lần lặp tối ưu bằng thực nghiệm
    """
    iterations_range = range(1, 10)
    results = {}
    
    for iters in iterations_range:
        qc = grover_3_qubit('111', iterations=iters)
        backend = Aer.get_backend('qasm_simulator')
        job = execute(qc, backend, shots=1000)
        result = job.result()
        counts = result.get_counts(qc)
        
        success_rate = counts.get('111', 0) / 1000
        results[iters] = success_rate
    
    return results
```

### Bài tập 2: Grover với nhiều giải pháp
```python
def grover_multiple_solutions(marked_states, iterations=1):
    """
    Thuật toán Grover với nhiều trạng thái được đánh dấu
    """
    n_qubits = len(marked_states[0])
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Khởi tạo
    for i in range(n_qubits):
        qc.h(i)
    
    # Oracle cho nhiều trạng thái
    for state in marked_states:
        oracle = grover_oracle(state)
        qc = qc.compose(oracle)
    
    # Diffusion
    diffusion = grover_diffusion(n_qubits)
    qc = qc.compose(diffusion)
    
    qc.measure(range(n_qubits), range(n_qubits))
    return qc
```

## 🎯 Kết Quả Mong Đợi
- Hiểu rõ nguyên lý hoạt động của thuật toán Grover
- Có thể triển khai thuật toán cho các bài toán tìm kiếm khác nhau
- Phân tích được hiệu suất và so sánh với phương pháp cổ điển

## 📖 Tài Liệu Tham Khảo
- [Qiskit Grover Algorithm](https://qiskit.org/textbook/ch-algorithms/grover.html)
- [Quantum Search Algorithms](https://qiskit.org/documentation/tutorials/algorithms/06_grover.html) 