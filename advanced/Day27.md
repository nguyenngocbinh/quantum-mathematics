# Day 27: Quantum Machine Learning Basics

## 🎯 Mục tiêu
- Hiểu nguyên lý cơ bản của Quantum Machine Learning
- Triển khai quantum feature maps và quantum kernels
- Xây dựng variational quantum circuits
- Áp dụng hybrid quantum-classical algorithms

## 🧠 Quantum Machine Learning - Tổng Quan

### Tại sao Quantum ML?
- **Quantum advantage**: Xử lý dữ liệu trong không gian Hilbert lớn
- **Feature mapping**: Chuyển đổi dữ liệu sang không gian lượng tử
- **Kernel methods**: Tính toán kernel functions hiệu quả
- **Hybrid algorithms**: Kết hợp ưu điểm của classical và quantum

```python
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.algorithms.optimizers import SPSA
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.algorithms import VQC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
```

## 🔧 Quantum Feature Maps

### 1. ZZFeatureMap - Feature Mapping Cơ Bản

```python
def quantum_feature_map_demo():
    """
    Demo quantum feature map với dữ liệu 2D
    """
    # Tạo dữ liệu mẫu
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                             n_informative=2, random_state=42, n_clusters_per_class=1)
    
    # Tạo quantum feature map
    feature_map = ZZFeatureMap(feature_dimension=2, reps=2)
    
    print("Quantum Feature Map:")
    print(feature_map)
    
    # Chuyển đổi dữ liệu sang quantum state
    backend = Aer.get_backend('statevector_simulator')
    
    quantum_states = []
    for i in range(min(5, len(X))):  # Chỉ xem 5 mẫu đầu
        # Bind parameters
        bound_circuit = feature_map.bind_parameters(X[i])
        
        # Execute
        job = execute(bound_circuit, backend)
        result = job.result()
        statevector = result.get_statevector()
        
        quantum_states.append(statevector)
        print(f"Sample {i}: {X[i]} -> State vector shape: {statevector.shape}")
    
    return quantum_states, feature_map

# Chạy demo
quantum_states, feature_map = quantum_feature_map_demo()
```

### 2. Custom Feature Map

```python
def custom_feature_map(n_features, reps=1):
    """
    Tạo custom quantum feature map
    """
    qc = QuantumCircuit(n_features)
    
    for rep in range(reps):
        # Encoding layer
        for i in range(n_features):
            qc.ry(Parameter(f'x_{i}'), i)
        
        # Entangling layer
        for i in range(n_features - 1):
            qc.cx(i, i + 1)
        qc.cx(n_features - 1, 0)
        
        # Rotation layer
        for i in range(n_features):
            qc.rz(Parameter(f'z_{i}'), i)
    
    return qc

# Tạo custom feature map
custom_map = custom_feature_map(2, reps=2)
print("Custom Feature Map:")
print(custom_map)
```

## 🎯 Quantum Kernels

### 1. Quantum Kernel với ZZFeatureMap

```python
def quantum_kernel_demo():
    """
    Demo quantum kernel classification
    """
    # Tạo dữ liệu
    X, y = make_classification(n_samples=50, n_features=2, n_redundant=0, 
                             n_informative=2, random_state=42, n_clusters_per_class=1)
    
    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Tạo quantum kernel
    feature_map = ZZFeatureMap(feature_dimension=2, reps=2)
    quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=Aer.get_backend('qasm_simulator'))
    
    # Train SVM với quantum kernel
    svm = SVC(kernel=quantum_kernel.evaluate)
    svm.fit(X_train, y_train)
    
    # Predict
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Quantum Kernel SVM Accuracy: {accuracy:.3f}")
    
    return svm, quantum_kernel, accuracy

# Chạy quantum kernel demo
svm_model, q_kernel, acc = quantum_kernel_demo()
```

### 2. Kernel Matrix Visualization

```python
def visualize_kernel_matrix(X, quantum_kernel):
    """
    Trực quan hóa kernel matrix
    """
    # Tính kernel matrix
    kernel_matrix = quantum_kernel.evaluate(X, X)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(kernel_matrix, cmap='viridis')
    plt.colorbar()
    plt.title('Quantum Kernel Matrix')
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')
    plt.show()
    
    return kernel_matrix

# Tạo dữ liệu và visualize
X_sample, _ = make_classification(n_samples=20, n_features=2, random_state=42)
feature_map = ZZFeatureMap(feature_dimension=2, reps=1)
q_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=Aer.get_backend('qasm_simulator'))
kernel_matrix = visualize_kernel_matrix(X_sample, q_kernel)
```

## 🔄 Variational Quantum Circuits (VQC)

### 1. VQC Classifier Cơ Bản

```python
def vqc_classifier_demo():
    """
    Demo VQC classifier
    """
    # Tạo dữ liệu
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                             n_informative=2, random_state=42, n_clusters_per_class=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Tạo feature map và ansatz
    feature_map = ZZFeatureMap(feature_dimension=2, reps=1)
    ansatz = RealAmplitudes(2, reps=2)
    
    # Tạo VQC
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=SPSA(maxiter=100),
        quantum_instance=Aer.get_backend('qasm_simulator')
    )
    
    # Train
    print("Training VQC...")
    vqc.fit(X_train, y_train)
    
    # Predict
    y_pred = vqc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"VQC Accuracy: {accuracy:.3f}")
    
    return vqc, accuracy

# Chạy VQC demo
vqc_model, vqc_acc = vqc_classifier_demo()
```

### 2. Custom Ansatz

```python
def custom_ansatz(n_qubits, depth=2):
    """
    Tạo custom ansatz cho VQC
    """
    qc = QuantumCircuit(n_qubits)
    
    for layer in range(depth):
        # Rotation layer
        for i in range(n_qubits):
            qc.ry(Parameter(f'ry_{layer}_{i}'), i)
            qc.rz(Parameter(f'rz_{layer}_{i}'), i)
        
        # Entangling layer
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        qc.cx(n_qubits - 1, 0)
    
    return qc

# Tạo custom ansatz
custom_ansatz_circuit = custom_ansatz(2, depth=3)
print("Custom Ansatz:")
print(custom_ansatz_circuit)
```

## 🎯 Hybrid Quantum-Classical Algorithms

### 1. Quantum-Classical Optimization

```python
def hybrid_optimization_demo():
    """
    Demo hybrid quantum-classical optimization
    """
    # Objective function: f(x) = x^2 + sin(x)
    def objective_function(x):
        return x**2 + np.sin(x)
    
    # Quantum-enhanced optimization
    def quantum_objective(params):
        # Simulate quantum computation
        qc = QuantumCircuit(1)
        qc.ry(params[0], 0)
        qc.measure_all()
        
        backend = Aer.get_backend('qasm_simulator')
        job = execute(qc, backend, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        # Map quantum result to classical objective
        quantum_value = counts.get('0', 0) / 1000  # Probability of |0⟩
        return objective_function(quantum_value * 2 * np.pi)
    
    # Optimize
    optimizer = SPSA(maxiter=50)
    initial_params = [0.5]
    
    result = optimizer.minimize(quantum_objective, initial_params)
    
    print(f"Optimal parameters: {result.x}")
    print(f"Optimal value: {result.fun}")
    
    return result

# Chạy hybrid optimization
opt_result = hybrid_optimization_demo()
```

### 2. Quantum Neural Network

```python
def quantum_neural_network():
    """
    Simple quantum neural network implementation
    """
    class QuantumNeuralNetwork:
        def __init__(self, n_qubits, n_layers):
            self.n_qubits = n_qubits
            self.n_layers = n_layers
            self.parameters = np.random.rand(n_layers * n_qubits * 3) * 2 * np.pi
        
        def create_circuit(self, x, params):
            qc = QuantumCircuit(self.n_qubits, 1)
            
            # Encode input
            for i in range(self.n_qubits):
                qc.ry(x[i] if i < len(x) else 0, i)
            
            # Variational layers
            param_idx = 0
            for layer in range(self.n_layers):
                # Rotation gates
                for i in range(self.n_qubits):
                    qc.ry(params[param_idx], i)
                    param_idx += 1
                    qc.rz(params[param_idx], i)
                    param_idx += 1
                    qc.rx(params[param_idx], i)
                    param_idx += 1
                
                # Entangling gates
                for i in range(self.n_qubits - 1):
                    qc.cx(i, i + 1)
                qc.cx(self.n_qubits - 1, 0)
            
            # Measure
            qc.measure(0, 0)
            return qc
        
        def forward(self, x):
            qc = self.create_circuit(x, self.parameters)
            backend = Aer.get_backend('qasm_simulator')
            job = execute(qc, backend, shots=1000)
            result = job.result()
            counts = result.get_counts()
            return counts.get('0', 0) / 1000  # Probability of |0⟩
    
    return QuantumNeuralNetwork

# Tạo QNN
QNN = quantum_neural_network()
qnn = QNN(n_qubits=2, n_layers=2)

# Test
test_input = [0.5, 0.3]
output = qnn.forward(test_input)
print(f"QNN output for {test_input}: {output:.3f}")
```

## 📊 So Sánh Hiệu Suất

```python
def compare_classical_quantum():
    """
    So sánh hiệu suất classical vs quantum ML
    """
    # Tạo dữ liệu
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                             n_informative=2, random_state=42, n_clusters_per_class=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Classical SVM
    classical_svm = SVC(kernel='rbf')
    classical_svm.fit(X_train, y_train)
    classical_acc = accuracy_score(y_test, classical_svm.predict(X_test))
    
    # Quantum SVM
    feature_map = ZZFeatureMap(feature_dimension=2, reps=1)
    quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=Aer.get_backend('qasm_simulator'))
    quantum_svm = SVC(kernel=quantum_kernel.evaluate)
    quantum_svm.fit(X_train, y_train)
    quantum_acc = accuracy_score(y_test, quantum_svm.predict(X_test))
    
    print(f"Classical SVM Accuracy: {classical_acc:.3f}")
    print(f"Quantum SVM Accuracy: {quantum_acc:.3f}")
    
    return {
        'classical': classical_acc,
        'quantum': quantum_acc
    }

# Chạy so sánh
comparison = compare_classical_quantum()
```

## 📚 Bài Tập Thực Hành

### Bài tập 1: Tối ưu hóa Feature Map
```python
def optimize_feature_map():
    """
    Tối ưu hóa feature map cho dataset cụ thể
    """
    # Tạo dataset phức tạp hơn
    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, 
                             n_informative=2, random_state=42, n_clusters_per_class=2)
    
    # Test các feature map khác nhau
    reps_list = [1, 2, 3, 4]
    accuracies = []
    
    for reps in reps_list:
        feature_map = ZZFeatureMap(feature_dimension=2, reps=reps)
        quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=Aer.get_backend('qasm_simulator'))
        
        svm = SVC(kernel=quantum_kernel.evaluate)
        svm.fit(X, y)
        acc = accuracy_score(y, svm.predict(X))
        accuracies.append(acc)
        
        print(f"Reps {reps}: Accuracy {acc:.3f}")
    
    return reps_list, accuracies
```

### Bài tập 2: Multi-class Classification
```python
def multi_class_quantum():
    """
    Quantum ML cho bài toán multi-class
    """
    from sklearn.datasets import make_blobs
    from sklearn.multiclass import OneVsRestClassifier
    
    # Tạo dữ liệu 3 classes
    X, y = make_blobs(n_samples=150, centers=3, random_state=42)
    
    # Quantum one-vs-rest classifier
    feature_map = ZZFeatureMap(feature_dimension=2, reps=2)
    quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=Aer.get_backend('qasm_simulator'))
    
    quantum_ovr = OneVsRestClassifier(SVC(kernel=quantum_kernel.evaluate))
    quantum_ovr.fit(X, y)
    
    acc = accuracy_score(y, quantum_ovr.predict(X))
    print(f"Multi-class Quantum Accuracy: {acc:.3f}")
    
    return quantum_ovr, acc
```

## 🎯 Kết Quả Mong Đợi
- Hiểu rõ nguyên lý Quantum Machine Learning
- Có thể triển khai quantum kernels và VQC
- So sánh được hiệu suất classical vs quantum ML
- Áp dụng được hybrid algorithms

## 📖 Tài Liệu Tham Khảo
- [Qiskit Machine Learning](https://qiskit.org/ecosystem/machine-learning/)
- [Quantum Feature Maps](https://qiskit.org/textbook/ch-machine-learning/machine-learning-qiskit-pytorch.html)
- [Variational Quantum Circuits](https://qiskit.org/textbook/ch-applications/vqe-molecules.html) 