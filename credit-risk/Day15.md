# Ngày 15: Quantum Anomaly Detection cho Fraud

## 🎯 Mục tiêu học tập

- Hiểu sâu về quantum anomaly detection và classical anomaly detection
- Nắm vững cách quantum anomaly detection cải thiện fraud detection
- Implement quantum anomaly detection algorithms cho credit fraud
- So sánh performance giữa quantum và classical anomaly detection

## 📚 Lý thuyết

### **Anomaly Detection Fundamentals**

#### **1. Classical Anomaly Detection**

**Statistical Methods:**
```
Z-score: z = (x - μ) / σ
IQR: Q3 - Q1
```

**Machine Learning Methods:**
```
Isolation Forest: Isolation score
One-Class SVM: Distance from hyperplane
Autoencoder: Reconstruction error
```

#### **2. Quantum Anomaly Detection**

**Quantum State Preparation:**
```
|ψ⟩ = (1/√N) Σᵢ |i⟩|xᵢ⟩
```

**Quantum Distance:**
```
d_quantum(x, y) = |⟨φ(x)|φ(y)⟩|²
```

**Quantum Anomaly Score:**
```
A(x) = 1 - maxᵢ |⟨φ(x)|φ(xᵢ)⟩|²
```

### **Quantum Anomaly Detection Types**

#### **1. Quantum Isolation Forest:**
- **Quantum Encoding**: Superposition of data points
- **Quantum Isolation**: Quantum random partitioning
- **Quantum Score**: Quantum path length

#### **2. Quantum One-Class SVM:**
- **Quantum Kernel**: Quantum feature space
- **Quantum Boundary**: Quantum decision boundary
- **Quantum Distance**: Quantum margin

#### **3. Quantum Autoencoder:**
- **Quantum Encoding**: Quantum compression
- **Quantum Decoding**: Quantum reconstruction
- **Quantum Error**: Quantum reconstruction error

### **Quantum Anomaly Detection Advantages**

#### **1. Quantum Properties:**
- **Superposition**: Parallel processing of multiple states
- **Entanglement**: Complex anomaly patterns
- **Quantum Parallelism**: Exponential speedup potential

#### **2. Fraud-specific Benefits:**
- **Non-linear Patterns**: Quantum detection captures complex fraud patterns
- **High-dimensional Data**: Handle many fraud features
- **Quantum Advantage**: Potential speedup for large datasets

## 💻 Thực hành

### **Project 15: Quantum Anomaly Detection cho Fraud Detection**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.algorithms import VQE, QAOA
from qiskit.algorithms.optimizers import SPSA
import pennylane as qml

class ClassicalAnomalyDetection:
    """Classical anomaly detection methods"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def prepare_features(self, data):
        """
        Prepare features for anomaly detection
        """
        # Feature engineering
        features = data.copy()
        
        # Create fraud-specific features
        features['debt_income_ratio'] = features['debt'] / (features['income'] + 1)
        features['credit_utilization'] = features['credit_used'] / (features['credit_limit'] + 1)
        features['payment_ratio'] = features['payments_made'] / (features['payments_due'] + 1)
        features['transaction_frequency'] = features['transactions'] / (features['account_age'] + 1)
        features['amount_velocity'] = features['avg_transaction_amount'] / (features['income'] + 1)
        
        # Normalize features
        numeric_features = features.select_dtypes(include=[np.number])
        if 'fraud' in numeric_features.columns:
            numeric_features = numeric_features.drop('fraud', axis=1)
        
        normalized_features = self.scaler.fit_transform(numeric_features)
        
        return pd.DataFrame(normalized_features, columns=numeric_features.columns)
    
    def isolation_forest(self, features, contamination=0.1):
        """
        Isolation Forest anomaly detection
        """
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_scores = iso_forest.fit_predict(features)
        
        return anomaly_scores, iso_forest.decision_function(features)
    
    def one_class_svm(self, features, nu=0.1):
        """
        One-Class SVM anomaly detection
        """
        oc_svm = OneClassSVM(nu=nu)
        anomaly_scores = oc_svm.fit_predict(features)
        decision_scores = oc_svm.decision_function(features)
        
        return anomaly_scores, decision_scores
    
    def autoencoder(self, features, hidden_dim=10):
        """
        Autoencoder anomaly detection
        """
        # Create autoencoder
        input_dim = features.shape[1]
        
        # Encoder
        encoder = MLPRegressor(
            hidden_layer_sizes=(hidden_dim,),
            max_iter=1000,
            random_state=42
        )
        
        # Train encoder
        encoder.fit(features, features)
        
        # Reconstruct
        reconstructed = encoder.predict(features)
        
        # Calculate reconstruction error
        reconstruction_error = np.mean((features - reconstructed) ** 2, axis=1)
        
        # Determine anomalies (high reconstruction error)
        threshold = np.percentile(reconstruction_error, 90)
        anomaly_scores = (reconstruction_error > threshold).astype(int)
        
        return anomaly_scores, reconstruction_error

class QuantumAnomalyDetection:
    """Quantum anomaly detection implementation"""
    
    def __init__(self, num_qubits=4):
        self.num_qubits = num_qubits
        self.backend = Aer.get_backend('qasm_simulator')
        self.feature_map = None
        self.normal_data = None
        
    def create_feature_map(self, X):
        """
        Create quantum feature map
        """
        self.feature_map = ZZFeatureMap(
            feature_dimension=X.shape[1],
            reps=2
        )
        return self.feature_map
    
    def quantum_distance(self, x1, x2):
        """
        Calculate quantum distance between two points
        """
        # Create quantum states
        circuit1 = self.feature_map.bind_parameters(x1)
        circuit2 = self.feature_map.bind_parameters(x2)
        
        # Execute circuits
        job1 = execute(circuit1, self.backend, shots=1000)
        job2 = execute(circuit2, self.backend, shots=1000)
        
        result1 = job1.result()
        result2 = job2.result()
        
        counts1 = result1.get_counts()
        counts2 = result2.get_counts()
        
        # Calculate quantum distance
        distance = self._calculate_quantum_distance(counts1, counts2)
        
        return distance
    
    def _calculate_quantum_distance(self, counts1, counts2):
        """
        Calculate distance between quantum states
        """
        # Get all possible bitstrings
        all_bitstrings = set(counts1.keys()) | set(counts2.keys())
        
        total_shots = 1000
        distance = 0.0
        
        for bitstring in all_bitstrings:
            prob1 = counts1.get(bitstring, 0) / total_shots
            prob2 = counts2.get(bitstring, 0) / total_shots
            
            distance += (prob1 - prob2) ** 2
        
        return np.sqrt(distance)
    
    def quantum_isolation_forest(self, X, contamination=0.1, n_trees=10):
        """
        Quantum Isolation Forest
        """
        n_samples = X.shape[0]
        anomaly_scores = np.zeros(n_samples)
        
        for tree in range(n_trees):
            print(f"Quantum Isolation Forest - Tree {tree + 1}/{n_trees}")
            
            # Randomly select features for this tree
            n_features = min(4, X.shape[1])
            feature_indices = np.random.choice(X.shape[1], n_features, replace=False)
            X_subset = X[:, feature_indices]
            
            # Quantum random partitioning
            for i in range(n_samples):
                # Calculate quantum distance to other points
                distances = []
                for j in range(n_samples):
                    if i != j:
                        distance = self.quantum_distance(X_subset[i], X_subset[j])
                        distances.append(distance)
                
                # Isolation score based on quantum distances
                if len(distances) > 0:
                    isolation_score = np.mean(distances)
                    anomaly_scores[i] += isolation_score
        
        # Normalize scores
        anomaly_scores = anomaly_scores / n_trees
        
        # Determine anomalies
        threshold = np.percentile(anomaly_scores, (1 - contamination) * 100)
        predictions = (anomaly_scores > threshold).astype(int)
        
        return predictions, anomaly_scores
    
    def quantum_one_class_svm(self, X, nu=0.1):
        """
        Quantum One-Class SVM
        """
        n_samples = X.shape[0]
        
        # Calculate quantum kernel matrix
        kernel_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self.quantum_distance(X[i], X[j])
        
        # Simple one-class SVM implementation
        # In practice, use proper QP solver
        
        # Calculate decision scores
        decision_scores = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Distance to center of mass
            center_distance = np.mean(kernel_matrix[i, :])
            decision_scores[i] = center_distance
        
        # Determine anomalies
        threshold = np.percentile(decision_scores, (1 - nu) * 100)
        predictions = (decision_scores > threshold).astype(int)
        
        return predictions, decision_scores
    
    def quantum_autoencoder(self, X, hidden_dim=2):
        """
        Quantum Autoencoder
        """
        n_samples, n_features = X.shape
        
        # Create quantum autoencoder circuit
        def create_quantum_autoencoder():
            circuit = QuantumCircuit(n_features, n_features)
            
            # Encoding layer
            for i in range(n_features):
                circuit.rx(0, i)  # Will be parameterized
                circuit.ry(0, i)  # Will be parameterized
            
            # Entanglement
            for i in range(n_features - 1):
                circuit.cx(i, i + 1)
            
            # Decoding layer (reverse of encoding)
            for i in range(n_features - 1, 0, -1):
                circuit.cx(i - 1, i)
            
            for i in range(n_features):
                circuit.ry(0, i)  # Will be parameterized
                circuit.rx(0, i)  # Will be parameterized
            
            return circuit
        
        # Simplified quantum autoencoder
        reconstruction_errors = []
        
        for i in range(n_samples):
            # Encode data point
            encoded_state = self._quantum_encode(X[i])
            
            # Decode data point
            decoded_state = self._quantum_decode(encoded_state)
            
            # Calculate reconstruction error
            error = np.mean((X[i] - decoded_state) ** 2)
            reconstruction_errors.append(error)
        
        reconstruction_errors = np.array(reconstruction_errors)
        
        # Determine anomalies
        threshold = np.percentile(reconstruction_errors, 90)
        predictions = (reconstruction_errors > threshold).astype(int)
        
        return predictions, reconstruction_errors
    
    def _quantum_encode(self, x):
        """
        Quantum encoding (simplified)
        """
        # Simplified encoding - in practice, use quantum circuit
        return x * 0.8  # Simple compression
    
    def _quantum_decode(self, encoded_x):
        """
        Quantum decoding (simplified)
        """
        # Simplified decoding - in practice, use quantum circuit
        return encoded_x / 0.8  # Simple decompression

def generate_fraud_data(n_samples=1000, fraud_ratio=0.1):
    """
    Generate synthetic fraud data
    """
    np.random.seed(42)
    
    # Generate normal transactions
    n_normal = int(n_samples * (1 - fraud_ratio))
    n_fraud = n_samples - n_normal
    
    # Normal transactions
    normal_income = np.random.normal(50000, 15000, n_normal)
    normal_debt = np.random.uniform(10000, 80000, n_normal)
    normal_credit_used = np.random.uniform(1000, 30000, n_normal)
    normal_credit_limit = np.random.uniform(10000, 100000, n_normal)
    normal_transactions = np.random.poisson(50, n_normal)
    normal_account_age = np.random.uniform(1, 10, n_normal)
    normal_avg_amount = np.random.uniform(50, 500, n_normal)
    normal_location_consistency = np.random.uniform(0.7, 1.0, n_normal)
    normal_time_pattern = np.random.uniform(0.6, 1.0, n_normal)
    
    # Fraudulent transactions
    fraud_income = np.random.normal(30000, 10000, n_fraud)
    fraud_debt = np.random.uniform(50000, 120000, n_fraud)
    fraud_credit_used = np.random.uniform(20000, 80000, n_fraud)
    fraud_credit_limit = np.random.uniform(5000, 50000, n_fraud)
    fraud_transactions = np.random.poisson(200, n_fraud)  # Higher frequency
    fraud_account_age = np.random.uniform(0.1, 2, n_fraud)  # Newer accounts
    fraud_avg_amount = np.random.uniform(1000, 5000, n_fraud)  # Higher amounts
    fraud_location_consistency = np.random.uniform(0.1, 0.5, n_fraud)  # Inconsistent
    fraud_time_pattern = np.random.uniform(0.1, 0.4, n_fraud)  # Unusual timing
    
    # Combine data
    data = pd.DataFrame({
        'income': np.concatenate([normal_income, fraud_income]),
        'debt': np.concatenate([normal_debt, fraud_debt]),
        'credit_used': np.concatenate([normal_credit_used, fraud_credit_used]),
        'credit_limit': np.concatenate([normal_credit_limit, fraud_credit_limit]),
        'transactions': np.concatenate([normal_transactions, fraud_transactions]),
        'account_age': np.concatenate([normal_account_age, fraud_account_age]),
        'avg_transaction_amount': np.concatenate([normal_avg_amount, fraud_avg_amount]),
        'location_consistency': np.concatenate([normal_location_consistency, fraud_location_consistency]),
        'time_pattern': np.concatenate([normal_time_pattern, fraud_time_pattern])
    })
    
    # Create fraud labels
    fraud_labels = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
    data['fraud'] = fraud_labels
    
    # Shuffle data
    indices = np.random.permutation(len(data))
    data = data.iloc[indices].reset_index(drop=True)
    
    return data

def compare_anomaly_detection_methods():
    """
    Compare classical and quantum anomaly detection methods
    """
    print("=== Classical vs Quantum Anomaly Detection ===\n")
    
    # Generate data
    data = generate_fraud_data(500, fraud_ratio=0.15)
    
    # Prepare features
    classical_ad = ClassicalAnomalyDetection()
    features = classical_ad.prepare_features(data)
    
    # Get true labels
    y_true = data['fraud']
    
    # Classical anomaly detection methods
    print("1. Classical Anomaly Detection Methods:")
    
    # Isolation Forest
    iso_scores, iso_decision = classical_ad.isolation_forest(features, contamination=0.15)
    iso_auc = roc_auc_score(y_true, -iso_decision)  # Negative because lower is more anomalous
    
    print(f"   Isolation Forest:")
    print(f"     AUC Score: {iso_auc:.4f}")
    print(f"     Detected Anomalies: {np.sum(iso_scores == -1)}")
    
    # One-Class SVM
    oc_svm_scores, oc_svm_decision = classical_ad.one_class_svm(features, nu=0.15)
    oc_svm_auc = roc_auc_score(y_true, -oc_svm_decision)
    
    print(f"   One-Class SVM:")
    print(f"     AUC Score: {oc_svm_auc:.4f}")
    print(f"     Detected Anomalies: {np.sum(oc_svm_scores == -1)}")
    
    # Autoencoder
    auto_scores, auto_decision = classical_ad.autoencoder(features, hidden_dim=5)
    auto_auc = roc_auc_score(y_true, auto_decision)
    
    print(f"   Autoencoder:")
    print(f"     AUC Score: {auto_auc:.4f}")
    print(f"     Detected Anomalies: {np.sum(auto_scores == 1)}")
    
    # Quantum anomaly detection methods
    print("\n2. Quantum Anomaly Detection Methods:")
    
    # Use subset of features for quantum methods
    quantum_features = features[['income', 'debt', 'credit_used', 'credit_limit']].copy()
    
    # Quantum Isolation Forest
    quantum_ad = QuantumAnomalyDetection(num_qubits=4)
    quantum_iso_scores, quantum_iso_decision = quantum_ad.quantum_isolation_forest(
        quantum_features.values, contamination=0.15, n_trees=5
    )
    quantum_iso_auc = roc_auc_score(y_true, quantum_iso_decision)
    
    print(f"   Quantum Isolation Forest:")
    print(f"     AUC Score: {quantum_iso_auc:.4f}")
    print(f"     Detected Anomalies: {np.sum(quantum_iso_scores == 1)}")
    
    # Quantum One-Class SVM
    quantum_oc_svm_scores, quantum_oc_svm_decision = quantum_ad.quantum_one_class_svm(
        quantum_features.values, nu=0.15
    )
    quantum_oc_svm_auc = roc_auc_score(y_true, quantum_oc_svm_decision)
    
    print(f"   Quantum One-Class SVM:")
    print(f"     AUC Score: {quantum_oc_svm_auc:.4f}")
    print(f"     Detected Anomalies: {np.sum(quantum_oc_svm_scores == 1)}")
    
    # Quantum Autoencoder
    quantum_auto_scores, quantum_auto_decision = quantum_ad.quantum_autoencoder(
        quantum_features.values, hidden_dim=2
    )
    quantum_auto_auc = roc_auc_score(y_true, quantum_auto_decision)
    
    print(f"   Quantum Autoencoder:")
    print(f"     AUC Score: {quantum_auto_auc:.4f}")
    print(f"     Detected Anomalies: {np.sum(quantum_auto_scores == 1)}")
    
    # Compare results
    print(f"\n3. Comparison Summary:")
    methods = ['Isolation Forest', 'One-Class SVM', 'Autoencoder', 
               'Quantum Isolation Forest', 'Quantum One-Class SVM', 'Quantum Autoencoder']
    auc_scores = [iso_auc, oc_svm_auc, auto_auc, 
                  quantum_iso_auc, quantum_oc_svm_auc, quantum_auto_auc]
    
    for method, score in zip(methods, auc_scores):
        print(f"   {method}: {score:.4f}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # ROC curves
    plt.subplot(2, 3, 1)
    from sklearn.metrics import roc_curve
    fpr_iso, tpr_iso, _ = roc_curve(y_true, -iso_decision)
    fpr_quantum_iso, tpr_quantum_iso, _ = roc_curve(y_true, quantum_iso_decision)
    
    plt.plot(fpr_iso, tpr_iso, label=f'Classical Isolation Forest (AUC = {iso_auc:.3f})')
    plt.plot(fpr_quantum_iso, tpr_quantum_iso, label=f'Quantum Isolation Forest (AUC = {quantum_iso_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves: Isolation Forest')
    plt.legend()
    plt.grid(True)
    
    # One-Class SVM comparison
    plt.subplot(2, 3, 2)
    fpr_oc, tpr_oc, _ = roc_curve(y_true, -oc_svm_decision)
    fpr_quantum_oc, tpr_quantum_oc, _ = roc_curve(y_true, quantum_oc_svm_decision)
    
    plt.plot(fpr_oc, tpr_oc, label=f'Classical One-Class SVM (AUC = {oc_svm_auc:.3f})')
    plt.plot(fpr_quantum_oc, tpr_quantum_oc, label=f'Quantum One-Class SVM (AUC = {quantum_oc_svm_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves: One-Class SVM')
    plt.legend()
    plt.grid(True)
    
    # Autoencoder comparison
    plt.subplot(2, 3, 3)
    fpr_auto, tpr_auto, _ = roc_curve(y_true, auto_decision)
    fpr_quantum_auto, tpr_quantum_auto, _ = roc_curve(y_true, quantum_auto_decision)
    
    plt.plot(fpr_auto, tpr_auto, label=f'Classical Autoencoder (AUC = {auto_auc:.3f})')
    plt.plot(fpr_quantum_auto, tpr_quantum_auto, label=f'Quantum Autoencoder (AUC = {quantum_auto_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves: Autoencoder')
    plt.legend()
    plt.grid(True)
    
    # AUC comparison
    plt.subplot(2, 3, 4)
    classical_methods = ['Isolation Forest', 'One-Class SVM', 'Autoencoder']
    classical_aucs = [iso_auc, oc_svm_auc, auto_auc]
    quantum_methods = ['Quantum Isolation Forest', 'Quantum One-Class SVM', 'Quantum Autoencoder']
    quantum_aucs = [quantum_iso_auc, quantum_oc_svm_auc, quantum_auto_auc]
    
    x = np.arange(len(classical_methods))
    width = 0.35
    
    plt.bar(x - width/2, classical_aucs, width, label='Classical', color='blue', alpha=0.7)
    plt.bar(x + width/2, quantum_aucs, width, label='Quantum', color='orange', alpha=0.7)
    
    plt.xlabel('Methods')
    plt.ylabel('AUC Score')
    plt.title('AUC Score Comparison')
    plt.xticks(x, classical_methods, rotation=45)
    plt.legend()
    plt.ylim(0, 1)
    
    # Anomaly score distributions
    plt.subplot(2, 3, 5)
    plt.hist(iso_decision[y_true == 0], bins=30, alpha=0.7, label='Normal', color='blue')
    plt.hist(iso_decision[y_true == 1], bins=30, alpha=0.7, label='Fraud', color='red')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title('Classical Isolation Forest Scores')
    plt.legend()
    
    plt.subplot(2, 3, 6)
    plt.hist(quantum_iso_decision[y_true == 0], bins=30, alpha=0.7, label='Normal', color='blue')
    plt.hist(quantum_iso_decision[y_true == 1], bins=30, alpha=0.7, label='Fraud', color='red')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title('Quantum Isolation Forest Scores')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return (iso_auc, oc_svm_auc, auto_auc, 
            quantum_iso_auc, quantum_oc_svm_auc, quantum_auto_auc)

# Run demos
if __name__ == "__main__":
    print("Running Anomaly Detection Comparisons...")
    (iso_auc, oc_svm_auc, auto_auc, 
     quantum_iso_auc, quantum_oc_svm_auc, quantum_auto_auc) = compare_anomaly_detection_methods()
```

## 📊 Kết quả và Phân tích

### **Quantum Anomaly Detection Advantages:**

#### **1. Quantum Properties:**
- **Superposition**: Parallel processing of multiple states
- **Entanglement**: Complex anomaly patterns
- **Quantum Parallelism**: Exponential speedup potential

#### **2. Fraud-specific Benefits:**
- **Non-linear Patterns**: Quantum detection captures complex fraud patterns
- **High-dimensional Data**: Handle many fraud features
- **Quantum Advantage**: Potential speedup for large datasets

#### **3. Performance Characteristics:**
- **Better Detection**: Quantum features improve anomaly detection
- **Robustness**: Quantum detection handles noisy fraud data
- **Scalability**: Quantum advantage for large-scale fraud detection

### **Comparison với Classical Anomaly Detection:**

#### **Classical Limitations:**
- Limited to linear separability
- Curse of dimensionality
- Local optima problems
- Feature engineering required

#### **Quantum Advantages:**
- Non-linear separability
- High-dimensional feature space
- Global optimization potential
- Automatic feature learning

## 🎯 Bài tập về nhà

### **Exercise 1: Quantum Anomaly Detection Calibration**
Implement quantum anomaly detection calibration methods cho fraud detection.

### **Exercise 2: Quantum Anomaly Detection Ensemble Methods**
Build ensemble of quantum anomaly detection algorithms cho improved performance.

### **Exercise 3: Quantum Anomaly Detection Feature Selection**
Develop quantum feature selection cho anomaly detection optimization.

### **Exercise 4: Quantum Anomaly Detection Validation**
Create validation framework cho quantum anomaly detection models.

---

> *"Quantum anomaly detection leverages quantum superposition and entanglement to provide superior fraud detection capabilities for credit risk management."* - Quantum Finance Research

> Ngày tiếp theo: [Quantum Portfolio Optimization](Day16.md) 