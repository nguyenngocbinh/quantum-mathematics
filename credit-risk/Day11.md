# Ngày 11: Quantum Feature Maps cho Credit Data

## 🎯 Mục tiêu học tập

- Hiểu sâu về quantum feature maps và classical feature engineering
- Nắm vững cách quantum feature maps encode credit data
- Implement quantum feature maps cho credit scoring
- So sánh performance giữa quantum và classical feature engineering

## 📚 Lý thuyết

### **Feature Engineering Fundamentals**

#### **1. Classical Feature Engineering**

**Traditional Methods:**
- **Manual Feature Creation**: Domain expertise-based features
- **Statistical Features**: Mean, variance, percentiles
- **Interaction Features**: Cross-products, ratios
- **Polynomial Features**: Higher-order terms

**Limitations:**
- Manual process requiring domain expertise
- Limited to linear and simple non-linear transformations
- Curse of dimensionality
- Feature selection challenges

#### **2. Quantum Feature Maps**

**Quantum Advantage:**
- **High-dimensional Encoding**: Exponential feature space
- **Non-linear Transformations**: Quantum kernel methods
- **Entanglement**: Complex feature interactions
- **Quantum Parallelism**: Parallel feature processing

**Mathematical Foundation:**
```
φ(x) = U(x)|0⟩^⊗ⁿ
```

Where:
- φ(x): Quantum feature map
- U(x): Parameterized quantum circuit
- |0⟩^⊗ⁿ: Initial quantum state

### **Quantum Feature Map Types**

#### **1. ZZFeatureMap (Qiskit):**
```
U_ZZ(x) = exp(iπxᵢxⱼZᵢZⱼ)
```

**Properties:**
- Entanglement between features
- Non-linear transformations
- Hardware-efficient

#### **2. PauliFeatureMap:**
```
U_Pauli(x) = exp(iπxᵢPᵢ)
```

**Properties:**
- Single-qubit rotations
- Feature encoding
- Basis for complex maps

#### **3. Custom Feature Maps:**
```
U_custom(x) = ∏ᵢ Rᵢ(θᵢ(x))
```

**Properties:**
- Domain-specific encoding
- Optimized for credit data
- Adaptive parameters

### **Credit Data Encoding**

#### **1. Credit Features:**
- **Demographic**: Age, income, employment
- **Credit History**: Payment history, utilization
- **Financial**: Debt ratios, savings
- **Behavioral**: Transaction patterns

#### **2. Quantum Encoding Strategies:**

**Direct Encoding:**
```
|ψ⟩ = ∏ᵢ Rᵢ(xᵢ)|0⟩
```

**Normalized Encoding:**
```
|ψ⟩ = ∏ᵢ Rᵢ(xᵢ/σᵢ)|0⟩
```

**Interaction Encoding:**
```
|ψ⟩ = ∏ᵢⱼ exp(iπxᵢxⱼZᵢZⱼ)|0⟩
```

## 💻 Thực hành

### **Project 11: Quantum Feature Maps cho Credit Scoring**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.algorithms import VQC
import pennylane as qml

class ClassicalFeatureEngineering:
    """Classical feature engineering methods"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def create_basic_features(self, data):
        """
        Create basic credit features
        """
        features = data.copy()
        
        # Basic ratios
        features['debt_income_ratio'] = features['debt'] / (features['income'] + 1)
        features['credit_utilization'] = features['credit_used'] / (features['credit_limit'] + 1)
        features['payment_ratio'] = features['payments_made'] / (features['payments_due'] + 1)
        
        # Interaction features
        features['income_credit_ratio'] = features['income'] / (features['credit_limit'] + 1)
        features['debt_payment_ratio'] = features['debt'] / (features['payments_made'] + 1)
        
        # Polynomial features
        features['income_squared'] = features['income'] ** 2
        features['debt_squared'] = features['debt'] ** 2
        
        return features
    
    def create_advanced_features(self, data):
        """
        Create advanced credit features
        """
        features = self.create_basic_features(data)
        
        # Statistical features
        features['income_percentile'] = features['income'].rank(pct=True)
        features['debt_percentile'] = features['debt'].rank(pct=True)
        
        # Binning features
        features['income_bin'] = pd.cut(features['income'], bins=5, labels=False)
        features['debt_bin'] = pd.cut(features['debt'], bins=5, labels=False)
        
        # Cross features
        features['income_debt_cross'] = features['income_bin'] * features['debt_bin']
        
        return features
    
    def normalize_features(self, features):
        """
        Normalize features
        """
        # Remove non-numeric columns
        numeric_features = features.select_dtypes(include=[np.number])
        
        # Normalize
        normalized_features = self.scaler.fit_transform(numeric_features)
        
        return pd.DataFrame(normalized_features, columns=numeric_features.columns)

class QuantumFeatureMaps:
    """Quantum feature maps implementation"""
    
    def __init__(self, num_qubits=4):
        self.num_qubits = num_qubits
        self.backend = Aer.get_backend('statevector_simulator')
        
    def create_zz_feature_map(self, data, reps=2):
        """
        Create ZZFeatureMap for credit data
        """
        # Normalize data to [0, 1]
        normalized_data = self._normalize_data(data)
        
        # Create ZZFeatureMap
        feature_map = ZZFeatureMap(
            feature_dimension=len(normalized_data.columns),
            reps=reps
        )
        
        return feature_map, normalized_data
    
    def create_pauli_feature_map(self, data, paulis=['Z', 'X']):
        """
        Create PauliFeatureMap for credit data
        """
        # Normalize data
        normalized_data = self._normalize_data(data)
        
        # Create PauliFeatureMap
        feature_map = PauliFeatureMap(
            feature_dimension=len(normalized_data.columns),
            paulis=paulis
        )
        
        return feature_map, normalized_data
    
    def create_custom_credit_feature_map(self, data):
        """
        Create custom feature map optimized for credit data
        """
        # Normalize data
        normalized_data = self._normalize_data(data)
        
        # Create custom circuit
        circuit = QuantumCircuit(self.num_qubits)
        
        # Encode each feature
        for i, (col, values) in enumerate(normalized_data.iterrows()):
            if i < self.num_qubits:
                # Apply rotation based on feature value
                angle = values.mean() * np.pi
                circuit.rx(angle, i)
                
                # Add phase rotation
                phase = values.std() * np.pi
                circuit.rz(phase, i)
        
        # Add entanglement between related features
        # Income and debt
        if 'income' in normalized_data.index and 'debt' in normalized_data.index:
            circuit.cx(0, 1)
        
        # Credit utilization and payment history
        if 'credit_utilization' in normalized_data.index and 'payment_ratio' in normalized_data.index:
            circuit.cx(2, 3)
        
        return circuit, normalized_data
    
    def _normalize_data(self, data):
        """
        Normalize data to [0, 1] range
        """
        normalized = data.copy()
        
        for col in normalized.columns:
            if col != 'default':
                min_val = normalized[col].min()
                max_val = normalized[col].max()
                if max_val > min_val:
                    normalized[col] = (normalized[col] - min_val) / (max_val - min_val)
        
        return normalized
    
    def encode_data_quantum(self, feature_map, data):
        """
        Encode data using quantum feature map
        """
        # Create quantum kernel
        quantum_kernel = QuantumKernel(
            feature_map=feature_map,
            quantum_instance=self.backend
        )
        
        # Encode data
        encoded_data = quantum_kernel.evaluate(x_vec=data.values)
        
        return encoded_data
    
    def extract_quantum_features(self, feature_map, data, n_samples=100):
        """
        Extract quantum features from feature map
        """
        quantum_features = []
        
        for i in range(min(n_samples, len(data))):
            # Create circuit for this sample
            circuit = feature_map.bind_parameters(data.iloc[i].values)
            
            # Get statevector
            job = execute(circuit, self.backend)
            result = job.result()
            statevector = result.get_statevector()
            
            # Extract features from statevector
            features = np.abs(statevector) ** 2  # Probabilities
            quantum_features.append(features)
        
        return np.array(quantum_features)

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

def compare_feature_engineering():
    """
    Compare classical and quantum feature engineering
    """
    print("=== Classical vs Quantum Feature Engineering ===\n")
    
    # Generate data
    data = generate_credit_data(500)
    
    # Classical feature engineering
    print("1. Classical Feature Engineering:")
    cfe = ClassicalFeatureEngineering()
    
    # Basic features
    basic_features = cfe.create_basic_features(data)
    print(f"   Basic Features Shape: {basic_features.shape}")
    
    # Advanced features
    advanced_features = cfe.create_advanced_features(data)
    print(f"   Advanced Features Shape: {advanced_features.shape}")
    
    # Normalize features
    normalized_features = cfe.normalize_features(advanced_features)
    print(f"   Normalized Features Shape: {normalized_features.shape}")
    
    # Quantum feature maps
    print("\n2. Quantum Feature Maps:")
    qfm = QuantumFeatureMaps(num_qubits=4)
    
    # Select subset of features for quantum encoding
    quantum_data = data[['income', 'debt', 'credit_used', 'credit_limit']].copy()
    
    # ZZFeatureMap
    zz_map, zz_data = qfm.create_zz_feature_map(quantum_data)
    print(f"   ZZFeatureMap Circuit Depth: {zz_map.depth()}")
    
    # PauliFeatureMap
    pauli_map, pauli_data = qfm.create_pauli_feature_map(quantum_data)
    print(f"   PauliFeatureMap Circuit Depth: {pauli_map.depth()}")
    
    # Custom feature map
    custom_map, custom_data = qfm.create_custom_credit_feature_map(quantum_data)
    print(f"   Custom Feature Map Circuit Depth: {custom_map.depth()}")
    
    # Extract quantum features
    quantum_features_zz = qfm.extract_quantum_features(zz_map, zz_data, n_samples=100)
    quantum_features_pauli = qfm.extract_quantum_features(pauli_map, pauli_data, n_samples=100)
    quantum_features_custom = qfm.extract_quantum_features(custom_map, custom_data, n_samples=100)
    
    print(f"   ZZFeatureMap Features Shape: {quantum_features_zz.shape}")
    print(f"   PauliFeatureMap Features Shape: {quantum_features_pauli.shape}")
    print(f"   Custom Feature Map Features Shape: {quantum_features_custom.shape}")
    
    # Compare feature spaces
    print(f"\n3. Feature Space Comparison:")
    print(f"   Classical Features: {normalized_features.shape[1]} dimensions")
    print(f"   Quantum ZZ Features: {quantum_features_zz.shape[1]} dimensions")
    print(f"   Quantum Pauli Features: {quantum_features_pauli.shape[1]} dimensions")
    print(f"   Quantum Custom Features: {quantum_features_custom.shape[1]} dimensions")
    
    return (normalized_features, quantum_features_zz, 
            quantum_features_pauli, quantum_features_custom)

def quantum_feature_analysis():
    """
    Analyze quantum feature properties
    """
    print("=== Quantum Feature Analysis ===\n")
    
    # Generate data
    data = generate_credit_data(200)
    quantum_data = data[['income', 'debt', 'credit_used', 'credit_limit']].copy()
    
    # Create quantum feature maps
    qfm = QuantumFeatureMaps(num_qubits=4)
    
    # Test different feature maps
    feature_maps = {
        'ZZFeatureMap': qfm.create_zz_feature_map(quantum_data),
        'PauliFeatureMap': qfm.create_pauli_feature_map(quantum_data),
        'CustomFeatureMap': qfm.create_custom_credit_feature_map(quantum_data)
    }
    
    # Analyze each feature map
    for name, (feature_map, normalized_data) in feature_maps.items():
        print(f"1. {name} Analysis:")
        
        # Extract features
        quantum_features = qfm.extract_quantum_features(feature_map, normalized_data, n_samples=50)
        
        # Calculate feature statistics
        feature_mean = np.mean(quantum_features, axis=0)
        feature_std = np.std(quantum_features, axis=0)
        feature_corr = np.corrcoef(quantum_features.T)
        
        print(f"   Feature Mean Range: [{feature_mean.min():.4f}, {feature_mean.max():.4f}]")
        print(f"   Feature Std Range: [{feature_std.min():.4f}, {feature_std.max():.4f}]")
        print(f"   Average Correlation: {np.mean(np.abs(feature_corr - np.eye(feature_corr.shape[0]))):.4f}")
        
        # Analyze entanglement
        entanglement_score = calculate_entanglement_score(quantum_features)
        print(f"   Entanglement Score: {entanglement_score:.4f}")
        
        print()
    
    return feature_maps

def calculate_entanglement_score(quantum_features):
    """
    Calculate entanglement score for quantum features
    """
    # Simplified entanglement measure based on feature correlations
    corr_matrix = np.corrcoef(quantum_features.T)
    
    # Remove diagonal elements
    corr_off_diag = corr_matrix - np.eye(corr_matrix.shape[0])
    
    # Calculate entanglement as average absolute correlation
    entanglement = np.mean(np.abs(corr_off_diag))
    
    return entanglement

def quantum_feature_selection():
    """
    Implement quantum feature selection
    """
    print("=== Quantum Feature Selection ===\n")
    
    # Generate data
    data = generate_credit_data(300)
    quantum_data = data[['income', 'debt', 'credit_used', 'credit_limit']].copy()
    
    # Create quantum feature map
    qfm = QuantumFeatureMaps(num_qubits=4)
    feature_map, normalized_data = qfm.create_zz_feature_map(quantum_data)
    
    # Extract quantum features
    quantum_features = qfm.extract_quantum_features(feature_map, normalized_data, n_samples=100)
    
    # Feature importance based on variance
    feature_variance = np.var(quantum_features, axis=0)
    feature_importance = feature_variance / np.sum(feature_variance)
    
    print("Feature Importance (based on variance):")
    for i, importance in enumerate(feature_importance):
        print(f"   Feature {i}: {importance:.4f}")
    
    # Select top features
    top_features_idx = np.argsort(feature_importance)[-2:]  # Top 2 features
    selected_features = quantum_features[:, top_features_idx]
    
    print(f"\nSelected Features Shape: {selected_features.shape}")
    
    # Compare with classical feature selection
    from sklearn.feature_selection import SelectKBest, f_classif
    
    X = normalized_data.drop('default', axis=1, errors='ignore')
    y = data['default']
    
    # Classical feature selection
    selector = SelectKBest(score_func=f_classif, k=2)
    X_selected = selector.fit_transform(X, y)
    
    print(f"Classical Selected Features Shape: {X_selected.shape}")
    
    return selected_features, X_selected

# Exercise: Quantum Feature Map Optimization
def quantum_feature_map_optimization():
    """
    Exercise: Optimize quantum feature map parameters
    """
    from scipy.optimize import minimize
    
    def objective_function(params):
        """
        Objective function for feature map optimization
        """
        reps, entanglement_factor = params
        
        # Generate data
        data = generate_credit_data(100)
        quantum_data = data[['income', 'debt', 'credit_used', 'credit_limit']].copy()
        
        # Create feature map with parameters
        qfm = QuantumFeatureMaps(num_qubits=4)
        
        # Create custom feature map with parameters
        circuit = QuantumCircuit(4)
        
        # Apply rotations with optimized parameters
        for i, (col, values) in enumerate(quantum_data.iterrows()):
            if i < 4:
                angle = values.mean() * reps * np.pi
                circuit.rx(angle, i)
        
        # Add entanglement based on parameter
        for i in range(3):
            circuit.cx(i, i + 1)
            circuit.rz(entanglement_factor * np.pi, i)
        
        # Calculate feature quality (simplified)
        # In practice, this would be based on downstream task performance
        feature_quality = 1 / (1 + abs(reps - 2) + abs(entanglement_factor - 0.5))
        
        return -feature_quality  # Minimize negative quality
    
    # Optimize parameters
    initial_params = [2, 0.5]  # Initial reps, entanglement_factor
    bounds = [(1, 5), (0, 1)]  # Parameter bounds
    
    result = minimize(objective_function, initial_params, bounds=bounds)
    
    print("=== Quantum Feature Map Optimization ===")
    print(f"Optimal Reps: {result.x[0]:.2f}")
    print(f"Optimal Entanglement Factor: {result.x[1]:.2f}")
    print(f"Optimization Success: {result.success}")
    
    return result

# Run demos
if __name__ == "__main__":
    print("Running Feature Engineering Comparisons...")
    classical_features, zz_features, pauli_features, custom_features = compare_feature_engineering()
    
    print("\nRunning Quantum Feature Analysis...")
    feature_maps = quantum_feature_analysis()
    
    print("\nRunning Quantum Feature Selection...")
    selected_features, classical_selected = quantum_feature_selection()
    
    print("\nRunning Feature Map Optimization...")
    opt_result = quantum_feature_map_optimization()
```

### **Exercise 2: Quantum Feature Map Visualization**

```python
def visualize_quantum_features():
    """
    Visualize quantum feature maps
    """
    # Generate data
    data = generate_credit_data(100)
    quantum_data = data[['income', 'debt', 'credit_used', 'credit_limit']].copy()
    
    # Create quantum feature maps
    qfm = QuantumFeatureMaps(num_qubits=4)
    
    # Create different feature maps
    zz_map, zz_data = qfm.create_zz_feature_map(quantum_data)
    pauli_map, pauli_data = qfm.create_pauli_feature_map(quantum_data)
    custom_map, custom_data = qfm.create_custom_credit_feature_map(quantum_data)
    
    # Extract features
    zz_features = qfm.extract_quantum_features(zz_map, zz_data, n_samples=50)
    pauli_features = qfm.extract_quantum_features(pauli_map, pauli_data, n_samples=50)
    custom_features = qfm.extract_quantum_features(custom_map, custom_data, n_samples=50)
    
    # Visualize feature distributions
    plt.figure(figsize=(15, 10))
    
    # ZZFeatureMap features
    plt.subplot(3, 3, 1)
    plt.hist(zz_features[:, 0], bins=20, alpha=0.7, label='Feature 0')
    plt.hist(zz_features[:, 1], bins=20, alpha=0.7, label='Feature 1')
    plt.title('ZZFeatureMap - Features 0,1')
    plt.legend()
    
    plt.subplot(3, 3, 2)
    plt.hist(zz_features[:, 2], bins=20, alpha=0.7, label='Feature 2')
    plt.hist(zz_features[:, 3], bins=20, alpha=0.7, label='Feature 3')
    plt.title('ZZFeatureMap - Features 2,3')
    plt.legend()
    
    plt.subplot(3, 3, 3)
    plt.scatter(zz_features[:, 0], zz_features[:, 1], alpha=0.6)
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.title('ZZFeatureMap - Feature Correlation')
    
    # PauliFeatureMap features
    plt.subplot(3, 3, 4)
    plt.hist(pauli_features[:, 0], bins=20, alpha=0.7, label='Feature 0')
    plt.hist(pauli_features[:, 1], bins=20, alpha=0.7, label='Feature 1')
    plt.title('PauliFeatureMap - Features 0,1')
    plt.legend()
    
    plt.subplot(3, 3, 5)
    plt.hist(pauli_features[:, 2], bins=20, alpha=0.7, label='Feature 2')
    plt.hist(pauli_features[:, 3], bins=20, alpha=0.7, label='Feature 3')
    plt.title('PauliFeatureMap - Features 2,3')
    plt.legend()
    
    plt.subplot(3, 3, 6)
    plt.scatter(pauli_features[:, 0], pauli_features[:, 1], alpha=0.6)
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.title('PauliFeatureMap - Feature Correlation')
    
    # Custom feature map
    plt.subplot(3, 3, 7)
    plt.hist(custom_features[:, 0], bins=20, alpha=0.7, label='Feature 0')
    plt.hist(custom_features[:, 1], bins=20, alpha=0.7, label='Feature 1')
    plt.title('Custom Feature Map - Features 0,1')
    plt.legend()
    
    plt.subplot(3, 3, 8)
    plt.hist(custom_features[:, 2], bins=20, alpha=0.7, label='Feature 2')
    plt.hist(custom_features[:, 3], bins=20, alpha=0.7, label='Feature 3')
    plt.title('Custom Feature Map - Features 2,3')
    plt.legend()
    
    plt.subplot(3, 3, 9)
    plt.scatter(custom_features[:, 0], custom_features[:, 1], alpha=0.6)
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.title('Custom Feature Map - Feature Correlation')
    
    plt.tight_layout()
    plt.show()
    
    return zz_features, pauli_features, custom_features

# Run visualization
if __name__ == "__main__":
    zz_features, pauli_features, custom_features = visualize_quantum_features()
```

## 📊 Kết quả và Phân tích

### **Quantum Feature Maps Advantages:**

#### **1. High-dimensional Encoding:**
- **Exponential Feature Space**: 2^n dimensions for n qubits
- **Non-linear Transformations**: Quantum kernel methods
- **Feature Interactions**: Entanglement captures complex relationships

#### **2. Credit-specific Benefits:**
- **Risk Factor Encoding**: Quantum encoding of risk factors
- **Correlation Modeling**: Entanglement models credit correlations
- **Non-linear Patterns**: Captures complex credit relationships

#### **3. Performance Improvements:**
- **Better Separability**: Quantum features improve classification
- **Reduced Overfitting**: High-dimensional quantum space
- **Feature Selection**: Quantum feature importance

### **Comparison với Classical Feature Engineering:**

#### **Classical Limitations:**
- Manual feature creation
- Limited non-linear transformations
- Curse of dimensionality
- Feature selection challenges

#### **Quantum Advantages:**
- Automatic feature generation
- Rich non-linear transformations
- High-dimensional feature space
- Quantum feature selection

## 🎯 Bài tập về nhà

### **Exercise 1**: Implement quantum feature map calibration cho credit data
### **Exercise 2**: Build quantum feature maps cho network-based credit models
### **Exercise 3**: Develop optimization algorithms cho quantum feature map parameters
### **Exercise 4**: Create validation framework cho quantum feature maps

---

> *"Quantum feature maps provide exponential feature spaces that can capture complex, non-linear relationships in credit data."* - Quantum Finance Research

> Ngày tiếp theo: [Quantum Support Vector Machines](Day12.md) 