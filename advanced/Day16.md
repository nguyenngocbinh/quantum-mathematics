# Day 16: Cài đặt Qiskit và Môi trường Python

## 🎯 Mục tiêu
- Cài đặt Python và Qiskit
- Hiểu cấu trúc cơ bản của Qiskit
- Chạy chương trình lượng tử đầu tiên

## 📦 Cài đặt môi trường

### 1. Cài đặt Python
```bash
# Tải Python từ python.org (khuyến nghị Python 3.8+)
# Hoặc sử dụng Anaconda
conda create -n quantum python=3.9
conda activate quantum
```

### 2. Cài đặt Qiskit
```bash
pip install qiskit
pip install qiskit[visualization]
```

### 3. Kiểm tra cài đặt
```python
import qiskit
print(f"Qiskit version: {qiskit.__version__}")
```

## 🔧 Cấu trúc Qiskit cơ bản

### Các thành phần chính:
- **Qiskit Terra**: Cấu trúc cơ bản và thuật toán
- **Qiskit Aer**: Mô phỏng lượng tử
- **Qiskit Ignis**: Đo lường và hiệu chuẩn
- **Qiskit Aqua**: Thuật toán ứng dụng

## 💻 Chương trình đầu tiên

```python
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Tạo mạch lượng tử với 1 qubit
qc = QuantumCircuit(1, 1)

# Áp dụng cổng Hadamard (tạo siêu vị)
qc.h(0)

# Đo qubit
qc.measure(0, 0)

# Vẽ mạch
print("Mạch lượng tử:")
print(qc)

# Chạy mô phỏng
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1000)
result = job.result()

# Hiển thị kết quả
counts = result.get_counts(qc)
print(f"Kết quả: {counts}")

# Vẽ biểu đồ
plot_histogram(counts)
plt.show()
```

## 📚 Bài tập thực hành

### Bài tập 1: Tạo qubit |0⟩ và |1⟩
```python
# Tạo qubit |0⟩
qc_0 = QuantumCircuit(1, 1)
qc_0.measure(0, 0)

# Tạo qubit |1⟩
qc_1 = QuantumCircuit(1, 1)
qc_1.x(0)  # Cổng NOT
qc_1.measure(0, 0)

# Chạy và so sánh kết quả
```

### Bài tập 2: Tạo siêu vị với góc khác nhau
```python
import numpy as np

def create_superposition(theta):
    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)  # Cổng quay Y
    qc.measure(0, 0)
    return qc

# Thử với các góc khác nhau
angles = [0, np.pi/4, np.pi/2, np.pi]
```

## 🎯 Kết quả mong đợi
- Hiểu cách cài đặt và sử dụng Qiskit
- Biết cách tạo mạch lượng tử đơn giản
- Có thể chạy mô phỏng và hiển thị kết quả

## 📖 Tài liệu tham khảo
- [Qiskit Documentation](https://qiskit.org/documentation/)
- [Qiskit Tutorials](https://qiskit.org/documentation/tutorials.html)
- [Python for Quantum Computing](https://qiskit.org/textbook/preface.html) 