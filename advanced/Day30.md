# Day 30: Quantum Neural Networks

## 🎯 Mục tiêu
- Hiểu nguyên lý hoạt động của Quantum Neural Networks (QNN)
- Triển khai quantum perceptron và quantum layers
- Áp dụng quantum gradient descent và backpropagation
- Xây dựng hybrid quantum-classical neural networks

## 🧠 Quantum Neural Networks - Tổng Quan

### Tại sao Quantum Neural Networks?
- **Quantum advantage**: Tận dụng quantum superposition và entanglement
- **Parameter space**: Khám phá không gian tham số lớn hơn
- **Feature mapping**: Quantum feature maps cho machine learning
- **Hybrid approach**: Kết hợp quantum và classical neural networks
- **Expressivity**: Khả năng biểu diễn phức tạp hơn classical networks

```python
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.algorithms.optimizers import SPSA, ADAM
from qiskit.quantum_info import Pauli
from qiskit.opflow import PauliSumOp, I, Z, X, Y
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

## 🔧 Quantum Neural Network Fundamentals

### 1. Quantum Perceptron

```python
def quantum_perceptron_demo():
    """
    Demo quantum perceptron cơ bản
    """
    # Quantum perceptron với 2 input qubits
    qc = QuantumCircuit(3)  # 2 input + 1 output
    
    # Input encoding
    x1 = Parameter('x₁')
    x2 = Parameter('x₂')
    
    # Weights
    w1 = Parameter('w₁')
    w2 = Parameter('w₂')
    b = Parameter('b')
    
    # Quantum perceptron circuit
    qc.rx(x1, 0)  # Encode input 1
    qc.rx(x2, 1)  # Encode input 2
    
    # Apply weights
    qc.rz(w1 * x1, 0)
    qc.rz(w2 * x2, 1)
    
    # Bias
    qc.rz(b, 2)
    
    # Entangling layer
    qc.cx(0, 2)
    qc.cx(1, 2)
    
    # Output measurement
    qc.measure(2, 0)
    
    return qc

# Test quantum perceptron
qnn_perceptron = quantum_perceptron_demo()
print("Quantum Perceptron Circuit:")
print(qnn_perceptron)
```

### 2. Quantum Feature Maps

```python
def quantum_feature_maps():
    """
    Các loại quantum feature maps khác nhau
    """
    
    def zz_feature_map(n_qubits, data_reps=2):
        """
        ZZ feature map: exp(i∑ᵢⱼ xᵢxⱼZᵢZⱼ)
        """
        qc = QuantumCircuit(n_qubits)
        
        # Data parameters
        x_params = [Parameter(f'x_{i}') for i in range(n_qubits)]
        
        # Apply Hadamard to all qubits
        for i in range(n_qubits):
            qc.h(i)
        
        # ZZ interactions
        for rep in range(data_reps):
            for i in range(n_qubits):
                for j in range(i+1, n_qubits):
                    qc.cx(i, j)
                    qc.rz(x_params[i] * x_params[j], j)
                    qc.cx(i, j)
        
        return qc, x_params
    
    def rotation_feature_map(n_qubits):
        """
        Rotation feature map: Rz(x)Ry(x)
        """
        qc = QuantumCircuit(n_qubits)
        
        x_params = [Parameter(f'x_{i}') for i in range(n_qubits)]
        
        for i in range(n_qubits):
            qc.rz(x_params[i], i)
            qc.ry(x_params[i], i)
        
        return qc, x_params
    
    return zz_feature_map, rotation_feature_map

# Test feature maps
zz_map, rot_map = quantum_feature_maps()
zz_circuit, zz_params = zz_map(3)
rot_circuit, rot_params = rot_map(3)

print("ZZ Feature Map:")
print(zz_circuit)
print(f"Parameters: {len(zz_params)}")

print("\nRotation Feature Map:")
print(rot_circuit)
print(f"Parameters: {len(rot_params)}")
```

## 🧪 Quantum Neural Network Implementation

### 1. Simple QNN cho Classification

```python
def simple_qnn_classifier():
    """
    QNN đơn giản cho bài toán classification
    """
    def create_qnn_circuit(n_qubits=2):
        qc = QuantumCircuit(n_qubits)
        
        # Feature encoding
        x1, x2 = Parameter('x₁'), Parameter('x₂')
        qc.rx(x1, 0)
        qc.rx(x2, 1)
        
        # Variational layers
        theta1, theta2, theta3 = Parameter('θ₁'), Parameter('θ₂'), Parameter('θ₃')
        qc.ry(theta1, 0)
        qc.ry(theta2, 1)
        qc.cx(0, 1)
        qc.rz(theta3, 1)
        
        # Measurement
        qc.measure_all()
        
        return qc
    
    return create_qnn_circuit

def qnn_training_demo():
    """
    Demo training QNN
    """
    # Generate synthetic data
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                             n_informative=2, random_state=42, n_clusters_per_class=1)
    
    # Normalize data
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create QNN circuit
    qnn_circuit = simple_qnn_classifier()(2)
    
    print("QNN Circuit for Classification:")
    print(qnn_circuit)
    
    return X_train, X_test, y_train, y_test, qnn_circuit

# QNN training demo
X_train, X_test, y_train, y_test, qnn_circ = qnn_training_demo()
```

### 2. Quantum Gradient Descent

```python
def quantum_gradient_descent():
    """
    Triển khai quantum gradient descent
    """
    def parameter_shift_gradient(circuit, params, param_idx, epsilon=0.1):
        """
        Parameter shift rule cho quantum gradient
        """
        # Forward shift
        params_plus = params.copy()
        params_plus[param_idx] += epsilon
        
        # Backward shift
        params_minus = params.copy()
        params_minus[param_idx] -= epsilon
        
        # Calculate gradients using parameter shift rule
        backend = Aer.get_backend('qasm_simulator')
        
        # Forward evaluation
        circuit_plus = circuit.bind_parameters(params_plus)
        job_plus = execute(circuit_plus, backend, shots=1000)
        result_plus = job_plus.result().get_counts()
        
        # Backward evaluation
        circuit_minus = circuit.bind_parameters(params_minus)
        job_minus = execute(circuit_minus, backend, shots=1000)
        result_minus = job_minus.result().get_counts()
        
        # Calculate expectation values
        exp_plus = (result_plus.get('00', 0) - result_plus.get('11', 0)) / 1000
        exp_minus = (result_minus.get('00', 0) - result_minus.get('11', 0)) / 1000
        
        # Gradient
        gradient = (exp_plus - exp_minus) / (2 * epsilon)
        
        return gradient
    
    return parameter_shift_gradient

# Test quantum gradient
q_gradient = quantum_gradient_descent()
```

### 3. Hybrid Quantum-Classical Neural Network

```python
class HybridQNN(nn.Module):
    """
    Hybrid Quantum-Classical Neural Network
    """
    def __init__(self, input_size, hidden_size, output_size, n_qubits=2):
        super(HybridQNN, self).__init__()
        
        # Classical layers
        self.classical_layer1 = nn.Linear(input_size, hidden_size)
        self.classical_layer2 = nn.Linear(hidden_size, n_qubits)
        self.output_layer = nn.Linear(n_qubits, output_size)
        
        # Quantum circuit
        self.qnn_circuit = self.create_quantum_circuit(n_qubits)
        
        # Quantum parameters
        self.quantum_params = nn.Parameter(torch.randn(3))  # 3 variational parameters
        
    def create_quantum_circuit(self, n_qubits):
        """
        Tạo quantum circuit cho QNN
        """
        qc = QuantumCircuit(n_qubits)
        
        # Input encoding
        for i in range(n_qubits):
            qc.rx(Parameter(f'x_{i}'), i)
        
        # Variational layers
        theta1, theta2, theta3 = Parameter('θ₁'), Parameter('θ₂'), Parameter('θ₃')
        qc.ry(theta1, 0)
        qc.ry(theta2, 1)
        qc.cx(0, 1)
        qc.rz(theta3, 1)
        
        # Measurement
        qc.measure_all()
        
        return qc
    
    def quantum_forward(self, x):
        """
        Quantum forward pass
        """
        # Convert classical input to quantum parameters
        x_quantum = x.detach().numpy()
        
        # Bind parameters to circuit
        param_dict = {}
        for i in range(len(x_quantum)):
            param_dict[Parameter(f'x_{i}')] = x_quantum[i]
        
        # Add variational parameters
        param_dict[Parameter('θ₁')] = self.quantum_params[0].item()
        param_dict[Parameter('θ₂')] = self.quantum_params[1].item()
        param_dict[Parameter('θ₃')] = self.quantum_params[2].item()
        
        # Execute quantum circuit
        bound_circuit = self.qnn_circuit.bind_parameters(param_dict)
        backend = Aer.get_backend('qasm_simulator')
        job = execute(bound_circuit, backend, shots=1000)
        result = job.result().get_counts()
        
        # Convert measurement results to expectation values
        exp_val = (result.get('00', 0) - result.get('11', 0)) / 1000
        
        return torch.tensor([exp_val], requires_grad=True)
    
    def forward(self, x):
        """
        Forward pass của hybrid network
        """
        # Classical layers
        x = torch.relu(self.classical_layer1(x))
        x = torch.relu(self.classical_layer2(x))
        
        # Quantum layer
        quantum_output = self.quantum_forward(x)
        
        # Output layer
        output = self.output_layer(quantum_output)
        
        return output

def hybrid_qnn_demo():
    """
    Demo hybrid quantum-classical neural network
    """
    # Create model
    model = HybridQNN(input_size=4, hidden_size=8, output_size=2, n_qubits=2)
    
    # Generate data
    X = torch.randn(100, 4)
    y = torch.randint(0, 2, (100,))
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("Hybrid QNN Model:")
    print(model)
    
    return model, X, y, criterion, optimizer

# Hybrid QNN demo
hybrid_model, X_data, y_data, loss_fn, opt = hybrid_qnn_demo()
```

## 🧬 Advanced QNN Architectures

### 1. Quantum Convolutional Neural Network (QCNN)

```python
def quantum_convolutional_layer():
    """
    Quantum convolutional layer
    """
    def create_qconv_circuit(n_qubits, kernel_size=2):
        qc = QuantumCircuit(n_qubits)
        
        # Convolutional parameters
        conv_params = [Parameter(f'conv_{i}') for i in range(kernel_size * 2)]
        
        # Apply convolutional operations
        for i in range(n_qubits - kernel_size + 1):
            # Kernel application
            for j in range(kernel_size):
                qc.ry(conv_params[j], i + j)
                qc.rz(conv_params[kernel_size + j], i + j)
            
            # Entangling within kernel
            for j in range(kernel_size - 1):
                qc.cx(i + j, i + j + 1)
        
        return qc, conv_params
    
    return create_qconv_circuit

def qcnn_architecture():
    """
    QCNN architecture demo
    """
    # Create QCNN layers
    qconv = quantum_convolutional_layer()
    
    # Layer 1: 4 qubits, kernel size 2
    layer1, params1 = qconv(4, 2)
    
    # Layer 2: 3 qubits, kernel size 2
    layer2, params2 = qconv(3, 2)
    
    print("QCNN Architecture:")
    print("Layer 1:")
    print(layer1)
    print(f"Parameters: {len(params1)}")
    
    print("\nLayer 2:")
    print(layer2)
    print(f"Parameters: {len(params2)}")
    
    return layer1, layer2, params1, params2

# QCNN demo
qcnn_layer1, qcnn_layer2, qcnn_params1, qcnn_params2 = qcnn_architecture()
```

### 2. Quantum Recurrent Neural Network (QRNN)

```python
def quantum_recurrent_cell():
    """
    Quantum recurrent cell
    """
    def create_qrnn_circuit(n_qubits, hidden_size):
        qc = QuantumCircuit(n_qubits + hidden_size)
        
        # Input qubits
        input_qubits = list(range(n_qubits))
        hidden_qubits = list(range(n_qubits, n_qubits + hidden_size))
        
        # Input encoding
        input_params = [Parameter(f'input_{i}') for i in range(n_qubits)]
        for i, param in enumerate(input_params):
            qc.rx(param, input_qubits[i])
        
        # Hidden state parameters
        hidden_params = [Parameter(f'hidden_{i}') for i in range(hidden_size)]
        for i, param in enumerate(hidden_params):
            qc.ry(param, hidden_qubits[i])
        
        # Recurrent connections
        for i in range(n_qubits):
            for j in range(hidden_size):
                qc.cx(input_qubits[i], hidden_qubits[j])
        
        # Hidden state update
        update_params = [Parameter(f'update_{i}') for i in range(hidden_size)]
        for i, param in enumerate(update_params):
            qc.rz(param, hidden_qubits[i])
        
        return qc, input_params + hidden_params + update_params
    
    return create_qrnn_circuit

def qrnn_demo():
    """
    QRNN demo
    """
    qrnn_cell = quantum_recurrent_cell()
    circuit, params = qrnn_cell(n_qubits=2, hidden_size=3)
    
    print("QRNN Cell:")
    print(circuit)
    print(f"Total parameters: {len(params)}")
    
    return circuit, params

# QRNN demo
qrnn_circuit, qrnn_params = qrnn_demo()
```

## 📊 QNN Training và Optimization

### 1. Quantum Natural Gradient

```python
def quantum_natural_gradient():
    """
    Quantum Natural Gradient optimization
    """
    def fisher_information_matrix(circuit, params, epsilon=0.01):
        """
        Tính Fisher Information Matrix
        """
        n_params = len(params)
        fim = np.zeros((n_params, n_params))
        
        # Calculate FIM using parameter shift
        for i in range(n_params):
            for j in range(n_params):
                # Forward shifts
                params_plus_i = params.copy()
                params_plus_i[i] += epsilon
                params_plus_j = params.copy()
                params_plus_j[j] += epsilon
                params_plus_ij = params.copy()
                params_plus_ij[i] += epsilon
                params_plus_ij[j] += epsilon
                
                # Backward shifts
                params_minus_i = params.copy()
                params_minus_i[i] -= epsilon
                params_minus_j = params.copy()
                params_minus_j[j] -= epsilon
                params_minus_ij = params.copy()
                params_minus_ij[i] -= epsilon
                params_minus_ij[j] -= epsilon
                
                # Calculate FIM element
                # F_ij = Cov[∂ᵢlog p(x|θ), ∂ⱼlog p(x|θ)]
                # Simplified calculation for demo
                fim[i, j] = np.random.normal(0, 0.1)  # Placeholder
        
        return fim
    
    return fisher_information_matrix

# Quantum Natural Gradient
qng_fim = quantum_natural_gradient()
```

### 2. QNN với Error Mitigation

```python
def qnn_error_mitigation():
    """
    Error mitigation techniques cho QNN
    """
    def zero_noise_extrapolation(circuit, params, noise_levels=[0, 0.1, 0.2]):
        """
        Zero Noise Extrapolation
        """
        results = []
        
        for noise_level in noise_levels:
            # Simulate with different noise levels
            backend = Aer.get_backend('qasm_simulator')
            bound_circuit = circuit.bind_parameters(params)
            
            # Add noise (simplified)
            job = execute(bound_circuit, backend, shots=1000)
            result = job.result().get_counts()
            
            # Calculate expectation value
            exp_val = (result.get('00', 0) - result.get('11', 0)) / 1000
            results.append(exp_val)
        
        # Extrapolate to zero noise
        # Linear extrapolation for demo
        if len(results) >= 2:
            zero_noise_result = results[0] - (results[1] - results[0]) * noise_levels[0] / (noise_levels[1] - noise_levels[0])
        else:
            zero_noise_result = results[0]
        
        return zero_noise_result
    
    return zero_noise_extrapolation

# Error mitigation
qnn_zne = qnn_error_mitigation()
```

## 🎯 Bài tập thực hành

### Bài tập 1: QNN cho XOR Problem
```python
def xor_qnn_exercise():
    """
    Bài tập: Implement QNN cho XOR problem
    """
    # XOR data
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    
    # TODO: Design QNN circuit for XOR
    # TODO: Implement training loop
    # TODO: Achieve 100% accuracy
    pass
```

### Bài tập 2: Quantum Autoencoder
```python
def quantum_autoencoder_exercise():
    """
    Bài tập: Quantum Autoencoder
    """
    # TODO: Design quantum encoder circuit
    # TODO: Implement quantum decoder
    # TODO: Train autoencoder for data compression
    pass
```

### Bài tập 3: Quantum Generative Adversarial Network (QGAN)
```python
def qgan_exercise():
    """
    Bài tập: Quantum GAN
    """
    # TODO: Design quantum generator
    # TODO: Implement quantum discriminator
    # TODO: Train QGAN for data generation
    pass
```

## 📚 Tài liệu tham khảo

### Papers và Research:
- "Quantum Neural Networks" - Beer et al.
- "Quantum Machine Learning" - Biamonte et al.
- "Variational Quantum Algorithms" - Cerezo et al.

### Frameworks:
- PennyLane
- TensorFlow Quantum
- Qiskit Machine Learning

### Applications:
- Quantum chemistry
- Financial modeling
- Drug discovery
- Image recognition

---

## 🎯 Tổng kết Day 30

### Kỹ năng đạt được:
- ✅ Hiểu nguyên lý Quantum Neural Networks
- ✅ Thiết kế quantum perceptron và feature maps
- ✅ Triển khai quantum gradient descent
- ✅ Xây dựng hybrid quantum-classical networks
- ✅ Áp dụng advanced QNN architectures

### Kiến thức quan trọng:
- **Quantum feature maps**: Encoding classical data into quantum states
- **Parameter shift rule**: Calculating quantum gradients
- **Hybrid architectures**: Combining quantum and classical components
- **Quantum advantage**: When QNNs outperform classical NNs

### Chuẩn bị cho giai đoạn tiếp theo:
- Quantum machine learning applications
- Real-world quantum computing projects
- Career preparation in quantum computing

---

## 🚀 Kết thúc Lộ trình Nâng cao

Chúc mừng! Bạn đã hoàn thành lộ trình 30 ngày về Quantum Mathematics với Python & Qiskit. 

### Những gì bạn đã đạt được:
- **Lập trình lượng tử**: Thành thạo Python, Qiskit, và quantum algorithms
- **Lý thuyết nâng cao**: Hiểu sâu các thuật toán Grover, Shor, VQE, QAOA
- **Ứng dụng thực tế**: Quantum chemistry, machine learning, cryptography
- **Kỹ năng chuyên nghiệp**: Dự án hoàn chỉnh và portfolio

### Bước tiếp theo:
1. **Capstone Project**: Xây dựng dự án cuối khóa
2. **Portfolio**: Tạo portfolio online
3. **Cộng đồng**: Tham gia quantum computing communities
4. **Sự nghiệp**: Tìm việc làm trong lĩnh vực lượng tử

*"The future of computing is quantum, and you're now part of that future!"* 🎉 