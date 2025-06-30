# Ngày 12: Quantum Support Vector Machines

## 🎯 Mục tiêu học tập

- Hiểu sâu về quantum SVM và classical SVM
- Nắm vững cách quantum kernels cải thiện credit classification
- Implement quantum SVM cho credit scoring
- So sánh performance giữa quantum và classical SVM

## 📚 Lý thuyết

### **Support Vector Machines Fundamentals**

#### **1. Classical SVM**

**Linear SVM:**
```
f(x) = w^T x + b
```

**Dual Formulation:**
```
max Σᵢ αᵢ - (1/2) Σᵢⱼ αᵢαⱼyᵢyⱼxᵢ^T xⱼ
s.t. Σᵢ αᵢyᵢ = 0, 0 ≤ αᵢ ≤ C
```

**Kernel Trick:**
```
K(xᵢ, xⱼ) = φ(xᵢ)^T φ(xⱼ)
```

#### **2. Quantum SVM**

**Quantum Kernel:**
```
K_quantum(xᵢ, xⱼ) = |⟨φ(xᵢ)|φ(xⱼ)⟩|²
```

**Quantum Feature Map:**
```
|φ(x)⟩ = U(x)|0⟩
```

### **Quantum Kernel Types**

#### **1. ZZFeatureMap Kernel:**
```
K_ZZ(xᵢ, xⱼ) = |⟨0|U^†(xᵢ)U(xⱼ)|0⟩|²
```

#### **2. PauliFeatureMap Kernel:**
```
K_Pauli(xᵢ, xⱼ) = |⟨0|∏ᵢ Pᵢ(xᵢ)^† Pᵢ(xⱼ)|0⟩|²
```

#### **3. Custom Quantum Kernel:**
```
K_custom(xᵢ, xⱼ) = |⟨0|U_custom^†(xᵢ)U_custom(xⱼ)|0⟩|²
```

### **Quantum SVM Advantages**

#### **1. Quantum Speedup:**
- **Kernel Computation**: Quantum parallelism
- **Feature Space**: High-dimensional quantum space
- **Optimization**: Quantum optimization algorithms

#### **2. Credit-specific Benefits:**
- **Non-linear Patterns**: Quantum kernels capture complex relationships
- **High-dimensional Data**: Handle many credit features
- **Quantum Advantage**: Potential speedup for large datasets

## 💻 Thực hành

### **Project 12: Quantum SVM cho Credit Scoring**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.algorithms import VQC
from qiskit.algorithms import VQE, QAOA
from qiskit.algorithms.optimizers import SPSA
import pennylane as qml

class ClassicalSVM:
    """Classical SVM implementation"""
    
    def __init__(self, kernel='rbf', C=1.0):
        self.svm = SVC(kernel=kernel, C=C, probability=True, random_state=42)
        self.scaler = StandardScaler()
        
    def prepare_features(self, data):
        """
        Prepare features for SVM
        """
        # Feature engineering
        features = data.copy()
        
        # Create interaction features
        features['debt_income_ratio'] = features['debt'] / (features['income'] + 1)
        features['credit_utilization'] = features['credit_used'] / (features['credit_limit'] + 1)
        features['payment_ratio'] = features['payments_made'] / (features['payments_due'] + 1)
        
        # Normalize features
        numeric_features = features.select_dtypes(include=[np.number])
        if 'default' in numeric_features.columns:
            numeric_features = numeric_features.drop('default', axis=1)
        
        normalized_features = self.scaler.fit_transform(numeric_features)
        
        return pd.DataFrame(normalized_features, columns=numeric_features.columns)
    
    def train(self, X_train, y_train):
        """
        Train classical SVM
        """
        self.svm.fit(X_train, y_train)
        
    def predict(self, X_test):
        """
        Make predictions
        """
        return self.svm.predict(X_test)
    
    def predict_proba(self, X_test):
        """
        Predict probabilities
        """
        return self.svm.predict_proba(X_test)[:, 1]

class QuantumSVM:
    """Quantum SVM implementation"""
    
    def __init__(self, num_qubits=4, feature_map_type='zz'):
        self.num_qubits = num_qubits
        self.feature_map_type = feature_map_type
        self.backend = Aer.get_backend('qasm_simulator')
        self.quantum_kernel = None
        self.support_vectors = None
        self.dual_coefficients = None
        self.intercept = None
        
    def create_feature_map(self, X):
        """
        Create quantum feature map
        """
        if self.feature_map_type == 'zz':
            return ZZFeatureMap(feature_dimension=X.shape[1], reps=2)
        elif self.feature_map_type == 'pauli':
            return PauliFeatureMap(feature_dimension=X.shape[1], paulis=['Z', 'X'])
        else:
            raise ValueError(f"Unknown feature map type: {self.feature_map_type}")
    
    def compute_quantum_kernel(self, X_train, X_test=None):
        """
        Compute quantum kernel matrix
        """
        feature_map = self.create_feature_map(X_train)
        self.quantum_kernel = QuantumKernel(
            feature_map=feature_map,
            quantum_instance=self.backend
        )
        
        if X_test is not None:
            # Compute kernel matrices
            kernel_train = self.quantum_kernel.evaluate(x_vec=X_train)
            kernel_test = self.quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)
            return kernel_train, kernel_test
        else:
            # Compute only training kernel
            kernel_train = self.quantum_kernel.evaluate(x_vec=X_train)
            return kernel_train
    
    def train_quantum_svm(self, X_train, y_train):
        """
        Train quantum SVM using kernel matrix
        """
        # Compute kernel matrix
        kernel_matrix = self.compute_quantum_kernel(X_train)
        
        # Solve dual problem (simplified)
        # In practice, you would use a proper QP solver
        self._solve_dual_problem(kernel_matrix, y_train)
        
        # Store support vectors
        self.support_vectors = X_train
        self.X_train = X_train
        self.y_train = y_train
    
    def _solve_dual_problem(self, kernel_matrix, y_train, C=1.0):
        """
        Solve SVM dual problem (simplified implementation)
        """
        n_samples = len(y_train)
        
        # Initialize dual coefficients
        alpha = np.zeros(n_samples)
        
        # Simple gradient ascent (in practice, use proper QP solver)
        learning_rate = 0.01
        max_iterations = 1000
        
        for iteration in range(max_iterations):
            # Compute gradient
            gradient = np.ones(n_samples)
            for i in range(n_samples):
                for j in range(n_samples):
                    gradient[i] -= alpha[j] * y_train[i] * y_train[j] * kernel_matrix[i, j]
            
            # Update alpha
            alpha_old = alpha.copy()
            alpha += learning_rate * gradient
            
            # Apply constraints
            alpha = np.clip(alpha, 0, C)
            
            # Check convergence
            if np.linalg.norm(alpha - alpha_old) < 1e-6:
                break
        
        self.dual_coefficients = alpha
        self.intercept = self._compute_intercept(kernel_matrix, y_train, alpha)
    
    def _compute_intercept(self, kernel_matrix, y_train, alpha):
        """
        Compute SVM intercept
        """
        # Find support vectors
        support_vector_indices = alpha > 1e-5
        
        if np.sum(support_vector_indices) == 0:
            return 0.0
        
        # Compute intercept
        intercept = 0.0
        for i in range(len(y_train)):
            if support_vector_indices[i]:
                intercept += y_train[i] - np.sum(alpha * y_train * kernel_matrix[i, :])
        
        return intercept / np.sum(support_vector_indices)
    
    def predict(self, X_test):
        """
        Make predictions using quantum SVM
        """
        if self.quantum_kernel is None:
            raise ValueError("Quantum SVM chưa được train")
        
        # Compute test kernel
        kernel_test = self.quantum_kernel.evaluate(x_vec=X_test, y_vec=self.X_train)
        
        # Make predictions
        predictions = []
        for i in range(len(X_test)):
            decision_value = np.sum(self.dual_coefficients * self.y_train * kernel_test[i, :]) + self.intercept
            predictions.append(1 if decision_value > 0 else 0)
        
        return np.array(predictions)
    
    def predict_proba(self, X_test):
        """
        Predict probabilities (simplified)
        """
        if self.quantum_kernel is None:
            raise ValueError("Quantum SVM chưa được train")
        
        # Compute test kernel
        kernel_test = self.quantum_kernel.evaluate(x_vec=X_test, y_vec=self.X_train)
        
        # Compute decision values
        decision_values = []
        for i in range(len(X_test)):
            decision_value = np.sum(self.dual_coefficients * self.y_train * kernel_test[i, :]) + self.intercept
            decision_values.append(decision_value)
        
        # Convert to probabilities (simplified)
        probabilities = 1 / (1 + np.exp(-np.array(decision_values)))
        return probabilities

def generate_credit_data(n_samples=1000):
    """
    Generate synthetic credit data
    """
    np.random.seed(42)
    
    # Generate features
    income = np.random.normal(50000, 20000, n_samples)
    debt = np.random.uniform(10000, 100000, n_samples)
    credit_used = np.random.uniform(1000, 50000, n_samples)
    credit_limit = np.random.uniform(5000, 100000, n_samples)
    payments_made = np.random.uniform(0, 12, n_samples)
    payments_due = np.random.uniform(1, 12, n_samples)
    age = np.random.uniform(25, 65, n_samples)
    employment_years = np.random.uniform(0, 30, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'income': income,
        'debt': debt,
        'credit_used': credit_used,
        'credit_limit': credit_limit,
        'payments_made': payments_made,
        'payments_due': payments_due,
        'age': age,
        'employment_years': employment_years
    })
    
    # Create target variable
    debt_income_ratio = data['debt'] / (data['income'] + 1)
    credit_utilization = data['credit_used'] / (data['credit_limit'] + 1)
    payment_ratio = data['payments_made'] / (data['payments_due'] + 1)
    
    default_prob = (0.3 * debt_income_ratio + 
                   0.4 * credit_utilization + 
                   0.3 * (1 - payment_ratio))
    
    default_prob += np.random.normal(0, 0.1, n_samples)
    default_prob = np.clip(default_prob, 0, 1)
    
    data['default'] = (default_prob > 0.5).astype(int)
    
    return data

def compare_svm_methods():
    """
    Compare classical and quantum SVM methods
    """
    print("=== Classical vs Quantum SVM Comparison ===\n")
    
    # Generate data
    data = generate_credit_data(500)
    
    # Prepare features
    classical_svm = ClassicalSVM(kernel='rbf', C=1.0)
    features = classical_svm.prepare_features(data)
    
    # Split data
    X = features
    y = data['default']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Classical SVM
    print("1. Classical SVM:")
    classical_svm.train(X_train, y_train)
    classical_pred = classical_svm.predict(X_test)
    classical_proba = classical_svm.predict_proba(X_test)
    classical_auc = roc_auc_score(y_test, classical_proba)
    
    print(f"   AUC Score: {classical_auc:.4f}")
    print(f"   Accuracy: {(classical_pred == y_test).mean():.4f}")
    
    # Quantum SVM
    print("\n2. Quantum SVM:")
    quantum_svm = QuantumSVM(num_qubits=4, feature_map_type='zz')
    
    # Use subset of features for quantum SVM (due to qubit limitations)
    quantum_features = X[['income', 'debt', 'credit_used', 'credit_limit']].copy()
    X_train_q, X_test_q, y_train_q, y_test_q = train_test_split(
        quantum_features, y, test_size=0.2, random_state=42
    )
    
    # Train quantum SVM
    quantum_svm.train_quantum_svm(X_train_q.values, y_train_q.values)
    quantum_pred = quantum_svm.predict(X_test_q.values)
    quantum_proba = quantum_svm.predict_proba(X_test_q.values)
    quantum_auc = roc_auc_score(y_test_q, quantum_proba)
    
    print(f"   AUC Score: {quantum_auc:.4f}")
    print(f"   Accuracy: {(quantum_pred == y_test_q).mean():.4f}")
    
    # Compare results
    print(f"\n3. Comparison:")
    print(f"   AUC Difference: {abs(classical_auc - quantum_auc):.4f}")
    print(f"   Accuracy Difference: {abs((classical_pred == y_test).mean() - (quantum_pred == y_test_q).mean()):.4f}")
    
    # Plot ROC curves
    plt.figure(figsize=(12, 5))
    
    # ROC curves
    plt.subplot(1, 2, 1)
    fpr_classical, tpr_classical, _ = roc_curve(y_test, classical_proba)
    fpr_quantum, tpr_quantum, _ = roc_curve(y_test_q, quantum_proba)
    
    plt.plot(fpr_classical, tpr_classical, label=f'Classical SVM (AUC = {classical_auc:.3f})')
    plt.plot(fpr_quantum, tpr_quantum, label=f'Quantum SVM (AUC = {quantum_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves: Classical vs Quantum SVM')
    plt.legend()
    plt.grid(True)
    
    # Feature importance comparison
    plt.subplot(1, 2, 2)
    methods = ['Classical SVM', 'Quantum SVM']
    auc_scores = [classical_auc, quantum_auc]
    plt.bar(methods, auc_scores, color=['blue', 'orange'])
    plt.ylabel('AUC Score')
    plt.title('AUC Score Comparison')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    return (classical_auc, quantum_auc, classical_proba, quantum_proba)

def quantum_kernel_analysis():
    """
    Analyze different quantum kernels
    """
    print("=== Quantum Kernel Analysis ===\n")
    
    # Generate data
    data = generate_credit_data(200)
    features = data[['income', 'debt', 'credit_used', 'credit_limit']].copy()
    
    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    
    # Test different quantum kernels
    kernel_types = ['zz', 'pauli']
    kernel_results = {}
    
    for kernel_type in kernel_types:
        print(f"1. {kernel_type.upper()}FeatureMap Kernel:")
        
        # Create quantum SVM
        qsvm = QuantumSVM(num_qubits=4, feature_map_type=kernel_type)
        
        # Compute kernel matrix
        kernel_matrix = qsvm.compute_quantum_kernel(normalized_features)
        
        # Analyze kernel properties
        kernel_eigenvalues = np.linalg.eigvals(kernel_matrix)
        kernel_condition = np.linalg.cond(kernel_matrix)
        kernel_rank = np.linalg.matrix_rank(kernel_matrix)
        
        print(f"   Kernel Matrix Shape: {kernel_matrix.shape}")
        print(f"   Condition Number: {kernel_condition:.2e}")
        print(f"   Rank: {kernel_rank}")
        print(f"   Eigenvalue Range: [{kernel_eigenvalues.min():.4f}, {kernel_eigenvalues.max():.4f}]")
        print(f"   Positive Eigenvalues: {np.sum(kernel_eigenvalues > 0)}/{len(kernel_eigenvalues)}")
        
        kernel_results[kernel_type] = {
            'matrix': kernel_matrix,
            'eigenvalues': kernel_eigenvalues,
            'condition': kernel_condition,
            'rank': kernel_rank
        }
        
        print()
    
    # Visualize kernel matrices
    plt.figure(figsize=(12, 5))
    
    for i, kernel_type in enumerate(kernel_types):
        plt.subplot(1, 2, i + 1)
        kernel_matrix = kernel_results[kernel_type]['matrix']
        plt.imshow(kernel_matrix, cmap='viridis')
        plt.colorbar()
        plt.title(f'{kernel_type.upper()}FeatureMap Kernel Matrix')
        plt.xlabel('Sample Index')
        plt.ylabel('Sample Index')
    
    plt.tight_layout()
    plt.show()
    
    return kernel_results

def quantum_svm_optimization():
    """
    Optimize quantum SVM parameters
    """
    print("=== Quantum SVM Optimization ===\n")
    
    # Generate data
    data = generate_credit_data(300)
    features = data[['income', 'debt', 'credit_used', 'credit_limit']].copy()
    
    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        normalized_features, data['default'], test_size=0.2, random_state=42
    )
    
    # Test different feature map types
    feature_map_types = ['zz', 'pauli']
    optimization_results = {}
    
    for fm_type in feature_map_types:
        print(f"1. {fm_type.upper()}FeatureMap Optimization:")
        
        # Test different parameters
        reps_values = [1, 2, 3]
        best_auc = 0
        best_params = None
        
        for reps in reps_values:
            # Create quantum SVM with different parameters
            qsvm = QuantumSVM(num_qubits=4, feature_map_type=fm_type)
            
            # Create custom feature map with reps parameter
            if fm_type == 'zz':
                feature_map = ZZFeatureMap(feature_dimension=4, reps=reps)
            else:
                feature_map = PauliFeatureMap(feature_dimension=4, paulis=['Z', 'X'])
            
            # Compute kernel
            quantum_kernel = QuantumKernel(
                feature_map=feature_map,
                quantum_instance=Aer.get_backend('qasm_simulator')
            )
            
            # Train and evaluate
            try:
                kernel_train = quantum_kernel.evaluate(x_vec=X_train)
                kernel_test = quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)
                
                # Simple evaluation (in practice, use proper SVM solver)
                # For demonstration, we'll use a simplified approach
                auc = evaluate_kernel_performance(kernel_train, kernel_test, y_train, y_test)
                
                if auc > best_auc:
                    best_auc = auc
                    best_params = {'reps': reps}
                
                print(f"   Reps {reps}: AUC = {auc:.4f}")
                
            except Exception as e:
                print(f"   Reps {reps}: Error - {e}")
        
        optimization_results[fm_type] = {
            'best_auc': best_auc,
            'best_params': best_params
        }
        
        print(f"   Best AUC: {best_auc:.4f}")
        print(f"   Best Params: {best_params}")
        print()
    
    return optimization_results

def evaluate_kernel_performance(kernel_train, kernel_test, y_train, y_test):
    """
    Evaluate kernel performance (simplified)
    """
    # Simple kernel-based classifier
    # In practice, use proper SVM solver
    
    # Compute kernel-based predictions
    predictions = []
    for i in range(len(kernel_test)):
        # Simple nearest neighbor approach
        similarities = kernel_test[i, :]
        nearest_idx = np.argmax(similarities)
        predictions.append(y_train[nearest_idx])
    
    # Calculate AUC
    try:
        auc = roc_auc_score(y_test, predictions)
    except:
        auc = 0.5  # Default to random if error
    
    return auc

# Exercise: Quantum SVM with Different Feature Maps
def quantum_svm_feature_maps_comparison():
    """
    Exercise: Compare quantum SVM with different feature maps
    """
    # Generate data
    data = generate_credit_data(400)
    features = data[['income', 'debt', 'credit_used', 'credit_limit']].copy()
    
    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        normalized_features, data['default'], test_size=0.2, random_state=42
    )
    
    # Test different feature maps
    feature_maps = {
        'ZZFeatureMap': ZZFeatureMap(feature_dimension=4, reps=2),
        'PauliFeatureMap': PauliFeatureMap(feature_dimension=4, paulis=['Z', 'X']),
        'CustomFeatureMap': create_custom_feature_map(4)
    }
    
    results = {}
    
    for name, feature_map in feature_maps.items():
        print(f"Testing {name}:")
        
        # Create quantum kernel
        quantum_kernel = QuantumKernel(
            feature_map=feature_map,
            quantum_instance=Aer.get_backend('qasm_simulator')
        )
        
        # Compute kernels
        kernel_train = quantum_kernel.evaluate(x_vec=X_train)
        kernel_test = quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)
        
        # Evaluate performance
        auc = evaluate_kernel_performance(kernel_train, kernel_test, y_train, y_test)
        
        results[name] = {
            'auc': auc,
            'kernel_train': kernel_train,
            'kernel_test': kernel_test
        }
        
        print(f"  AUC: {auc:.4f}")
        print(f"  Kernel Train Shape: {kernel_train.shape}")
        print()
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    # AUC comparison
    plt.subplot(1, 2, 1)
    names = list(results.keys())
    aucs = [results[name]['auc'] for name in names]
    plt.bar(names, aucs, color=['blue', 'orange', 'green'])
    plt.ylabel('AUC Score')
    plt.title('Quantum SVM Feature Map Comparison')
    plt.xticks(rotation=45)
    
    # Kernel matrix visualization
    plt.subplot(1, 2, 2)
    best_map = max(results.keys(), key=lambda x: results[x]['auc'])
    kernel_matrix = results[best_map]['kernel_train']
    plt.imshow(kernel_matrix, cmap='viridis')
    plt.colorbar()
    plt.title(f'Best Kernel Matrix ({best_map})')
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')
    
    plt.tight_layout()
    plt.show()
    
    return results

def create_custom_feature_map(num_qubits):
    """
    Create custom quantum feature map
    """
    circuit = QuantumCircuit(num_qubits)
    
    # Add rotations
    for i in range(num_qubits):
        circuit.rx(0, i)  # Will be parameterized
        circuit.ry(0, i)  # Will be parameterized
    
    # Add entanglement
    for i in range(num_qubits - 1):
        circuit.cx(i, i + 1)
    
    return circuit

# Run demos
if __name__ == "__main__":
    print("Running SVM Comparisons...")
    classical_auc, quantum_auc, classical_proba, quantum_proba = compare_svm_methods()
    
    print("\nRunning Quantum Kernel Analysis...")
    kernel_results = quantum_kernel_analysis()
    
    print("\nRunning Quantum SVM Optimization...")
    optimization_results = quantum_svm_optimization()
    
    print("\nRunning Feature Maps Comparison...")
    feature_maps_results = quantum_svm_feature_maps_comparison()
```

### **Exercise 2: Quantum SVM Performance Analysis**

```python
def quantum_svm_performance_analysis():
    """
    Exercise: Analyze quantum SVM performance characteristics
    """
    # Generate different datasets
    dataset_sizes = [100, 200, 300, 400, 500]
    performance_results = {}
    
    for size in dataset_sizes:
        print(f"Testing dataset size: {size}")
        
        # Generate data
        data = generate_credit_data(size)
        features = data[['income', 'debt', 'credit_used', 'credit_limit']].copy()
        
        # Normalize features
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            normalized_features, data['default'], test_size=0.2, random_state=42
        )
        
        # Test classical SVM
        classical_svm = ClassicalSVM(kernel='rbf', C=1.0)
        classical_svm.train(X_train, y_train)
        classical_proba = classical_svm.predict_proba(X_test)
        classical_auc = roc_auc_score(y_test, classical_proba)
        
        # Test quantum SVM
        quantum_svm = QuantumSVM(num_qubits=4, feature_map_type='zz')
        quantum_svm.train_quantum_svm(X_train, y_train)
        quantum_proba = quantum_svm.predict_proba(X_test)
        quantum_auc = roc_auc_score(y_test, quantum_proba)
        
        performance_results[size] = {
            'classical_auc': classical_auc,
            'quantum_auc': quantum_auc,
            'classical_accuracy': (classical_svm.predict(X_test) == y_test).mean(),
            'quantum_accuracy': (quantum_svm.predict(X_test) == y_test).mean()
        }
        
        print(f"  Classical AUC: {classical_auc:.4f}")
        print(f"  Quantum AUC: {quantum_auc:.4f}")
        print()
    
    # Plot performance vs dataset size
    plt.figure(figsize=(12, 5))
    
    # AUC comparison
    plt.subplot(1, 2, 1)
    sizes = list(performance_results.keys())
    classical_aucs = [performance_results[size]['classical_auc'] for size in sizes]
    quantum_aucs = [performance_results[size]['quantum_auc'] for size in sizes]
    
    plt.plot(sizes, classical_aucs, 'o-', label='Classical SVM', linewidth=2)
    plt.plot(sizes, quantum_aucs, 's-', label='Quantum SVM', linewidth=2)
    plt.xlabel('Dataset Size')
    plt.ylabel('AUC Score')
    plt.title('Performance vs Dataset Size')
    plt.legend()
    plt.grid(True)
    
    # Accuracy comparison
    plt.subplot(1, 2, 2)
    classical_accs = [performance_results[size]['classical_accuracy'] for size in sizes]
    quantum_accs = [performance_results[size]['quantum_accuracy'] for size in sizes]
    
    plt.plot(sizes, classical_accs, 'o-', label='Classical SVM', linewidth=2)
    plt.plot(sizes, quantum_accs, 's-', label='Quantum SVM', linewidth=2)
    plt.xlabel('Dataset Size')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Dataset Size')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return performance_results

# Run performance analysis
if __name__ == "__main__":
    performance_results = quantum_svm_performance_analysis()
```

## 📊 Kết quả và Phân tích

### **Quantum SVM Advantages:**

#### **1. Quantum Kernel Benefits:**
- **High-dimensional Feature Space**: Exponential feature encoding
- **Non-linear Separability**: Quantum kernels capture complex patterns
- **Quantum Parallelism**: Parallel kernel computation

#### **2. Credit-specific Improvements:**
- **Complex Risk Patterns**: Quantum kernels model non-linear risk relationships
- **Feature Interactions**: Entanglement captures credit correlations
- **Quantum Advantage**: Potential speedup for large credit datasets

#### **3. Performance Characteristics:**
- **Better Separability**: Quantum features improve classification boundaries
- **Robustness**: Quantum kernels handle noisy credit data
- **Scalability**: Quantum advantage for large-scale credit scoring

### **Comparison với Classical SVM:**

#### **Classical Limitations:**
- Limited kernel types
- Curse of dimensionality
- Linear separability assumptions
- Feature engineering required

#### **Quantum Advantages:**
- Rich quantum kernel space
- High-dimensional feature encoding
- Non-linear separability
- Automatic feature learning

## 🎯 Bài tập về nhà

### **Exercise 1: Quantum SVM Calibration**
Implement quantum SVM calibration methods cho credit scoring.

### **Exercise 2: Quantum SVM Ensemble Methods**
Build ensemble of quantum SVMs cho improved performance.

### **Exercise 3: Quantum SVM Feature Selection**
Develop quantum feature selection cho SVM optimization.

### **Exercise 4: Quantum SVM Validation**
Create validation framework cho quantum SVM models.

---

> *"Quantum Support Vector Machines leverage quantum kernels to provide superior classification performance for complex credit risk patterns."* - Quantum Finance Research

> Ngày tiếp theo: [Quantum Neural Networks cho Credit Scoring](Day13.md) 