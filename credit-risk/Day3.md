# Ngày 3: Credit Scoring và Machine Learning

## 🎯 Mục tiêu học tập

- Hiểu sâu về credit scoring và các phương pháp machine learning truyền thống
- Phân tích hạn chế của classical ML trong credit scoring
- Implement quantum-enhanced credit scoring models
- So sánh performance giữa classical ML và quantum approaches

## 📚 Lý thuyết

### **Credit Scoring Fundamentals**

#### **1. Credit Scoring là gì?**
Credit scoring là quá trình đánh giá khả năng trả nợ của người vay dựa trên các thông tin tài chính và hành vi.

#### **2. Các loại Credit Score:**
- **FICO Score**: Industry standard (300-850)
- **VantageScore**: Alternative scoring model
- **Custom Scores**: Bank-specific models
- **Behavioral Scores**: Based on transaction patterns

#### **3. Key Features trong Credit Scoring:**
- **Demographic**: Age, income, employment
- **Credit History**: Payment history, credit utilization
- **Financial**: Debt-to-income ratio, savings
- **Behavioral**: Transaction patterns, spending habits

### **Classical Machine Learning cho Credit Scoring**

#### **1. Logistic Regression:**
```
P(default) = 1 / (1 + e^(-β₀ - β₁x₁ - ... - βₙxₙ))
```

**Ưu điểm:**
- Interpretable coefficients
- Fast training và prediction
- Well-understood statistical properties

**Nhược điểm:**
- Linear assumptions
- Limited feature interactions
- Sensitive to outliers

#### **2. Random Forest:**
- Ensemble of decision trees
- Handles non-linear relationships
- Feature importance ranking

#### **3. Gradient Boosting (XGBoost, LightGBM):**
- Sequential tree building
- High predictive performance
- Handles missing values well

#### **4. Neural Networks:**
- Deep learning cho complex patterns
- Feature learning capabilities
- High computational requirements

### **Hạn chế của Classical ML**

#### **1. Feature Engineering Challenges:**
- Manual feature creation
- Domain expertise required
- Limited feature interactions

#### **2. Interpretability Issues:**
- Black-box models
- Regulatory compliance challenges
- Difficulty explaining decisions

#### **3. Data Limitations:**
- Sparse default events
- Imbalanced datasets
- Missing data handling

#### **4. Computational Constraints:**
- Training time cho large datasets
- Real-time scoring requirements
- Model complexity vs performance trade-off

### **Quantum Advantages cho Credit Scoring**

#### **1. Quantum Feature Maps:**
- High-dimensional feature encoding
- Non-linear transformations
- Quantum kernel methods

#### **2. Quantum Neural Networks:**
- Quantum-enhanced neural networks
- Superposition-based learning
- Entanglement for feature interactions

#### **3. Quantum Support Vector Machines:**
- Quantum kernel computation
- High-dimensional classification
- Quantum speedup cho training

#### **4. Quantum Clustering:**
- Quantum k-means
- Quantum DBSCAN
- Anomaly detection

## 💻 Thực hành

### **Project 3: Quantum-Enhanced Credit Scoring System**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.algorithms import VQE, QAOA
from qiskit.algorithms.optimizers import SPSA
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.kernels import QuantumKernel
import pennylane as qml

class ClassicalCreditScoring:
    """Classical credit scoring models"""
    
    def __init__(self):
        self.logistic_model = LogisticRegression(random_state=42)
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def prepare_features(self, data):
        """
        Prepare features for credit scoring
        """
        # Feature engineering
        features = data.copy()
        
        # Create interaction features
        features['income_debt_ratio'] = features['income'] / (features['debt'] + 1)
        features['credit_utilization_ratio'] = features['credit_used'] / (features['credit_limit'] + 1)
        features['payment_ratio'] = features['payments_made'] / (features['payments_due'] + 1)
        
        # Normalize features
        for col in features.columns:
            if col != 'default':
                features[col] = (features[col] - features[col].mean()) / features[col].std()
        
        return features
    
    def train_models(self, X_train, y_train):
        """
        Train classical models
        """
        # Train Logistic Regression
        self.logistic_model.fit(X_train, y_train)
        
        # Train Random Forest
        self.rf_model.fit(X_train, y_train)
        
    def predict(self, X_test):
        """
        Make predictions
        """
        lr_pred = self.logistic_model.predict_proba(X_test)[:, 1]
        rf_pred = self.rf_model.predict_proba(X_test)[:, 1]
        
        return lr_pred, rf_pred

class QuantumCreditScoring:
    """Quantum-enhanced credit scoring"""
    
    def __init__(self, num_qubits=4):
        self.num_qubits = num_qubits
        self.backend = Aer.get_backend('qasm_simulator')
        self.quantum_kernel = None
        self.vqc_model = None
        
    def create_quantum_feature_map(self, X):
        """
        Create quantum feature map
        """
        # Use ZZFeatureMap for encoding
        feature_map = ZZFeatureMap(feature_dimension=X.shape[1], reps=2)
        return feature_map
    
    def create_quantum_kernel(self, X_train, X_test):
        """
        Create quantum kernel
        """
        feature_map = self.create_quantum_feature_map(X_train)
        self.quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=self.backend)
        
        # Compute kernel matrices
        kernel_train = self.quantum_kernel.evaluate(x_vec=X_train)
        kernel_test = self.quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)
        
        return kernel_train, kernel_test
    
    def create_vqc_model(self, X_train, y_train):
        """
        Create Variational Quantum Classifier
        """
        feature_map = self.create_quantum_feature_map(X_train)
        ansatz = RealAmplitudes(self.num_qubits, reps=2)
        
        self.vqc_model = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=SPSA(maxiter=100),
            quantum_instance=self.backend
        )
        
        # Train the model
        self.vqc_model.fit(X_train, y_train)
        
    def predict_quantum(self, X_test):
        """
        Make quantum predictions
        """
        if self.vqc_model is not None:
            return self.vqc_model.predict_proba(X_test)[:, 1]
        else:
            raise ValueError("VQC model chưa được train")

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
    credit_data = pd.DataFrame({
        'income': income,
        'debt': debt,
        'credit_used': credit_used,
        'credit_limit': credit_limit,
        'payments_made': payments_made,
        'payments_due': payments_due,
        'age': age,
        'employment_years': employment_years
    })
    
    # Create target variable (default probability)
    # Higher debt-to-income ratio increases default probability
    debt_income_ratio = credit_data['debt'] / (credit_data['income'] + 1)
    credit_utilization = credit_data['credit_used'] / (credit_data['credit_limit'] + 1)
    payment_ratio = credit_data['payments_made'] / (credit_data['payments_due'] + 1)
    
    # Calculate default probability
    default_prob = (0.3 * debt_income_ratio + 
                   0.4 * credit_utilization + 
                   0.3 * (1 - payment_ratio))
    
    # Add noise
    default_prob += np.random.normal(0, 0.1, n_samples)
    default_prob = np.clip(default_prob, 0, 1)
    
    # Create binary target
    credit_data['default'] = (default_prob > 0.5).astype(int)
    
    return credit_data

def compare_models():
    """
    Compare classical and quantum credit scoring models
    """
    # Generate data
    data = generate_credit_data(1000)
    
    # Prepare features
    classical_scoring = ClassicalCreditScoring()
    features = classical_scoring.prepare_features(data)
    
    # Split data
    X = features.drop('default', axis=1)
    y = features['default']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train classical models
    classical_scoring.train_models(X_train, y_train)
    lr_pred, rf_pred = classical_scoring.predict(X_test)
    
    # Train quantum model
    quantum_scoring = QuantumCreditScoring(num_qubits=4)
    quantum_scoring.create_vqc_model(X_train.values, y_train.values)
    quantum_pred = quantum_scoring.predict_quantum(X_test.values)
    
    # Evaluate models
    models = {
        'Logistic Regression': lr_pred,
        'Random Forest': rf_pred,
        'Quantum VQC': quantum_pred
    }
    
    results = {}
    for name, predictions in models.items():
        auc = roc_auc_score(y_test, predictions)
        results[name] = auc
        print(f"{name} - AUC: {auc:.4f}")
    
    # Plot ROC curves
    plt.figure(figsize=(10, 6))
    for name, predictions in models.items():
        fpr, tpr, _ = roc_curve(y_test, predictions)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {results[name]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves: Classical vs Quantum Credit Scoring')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return results

# Exercise: Quantum Feature Engineering
def quantum_feature_engineering_exercise():
    """
    Exercise: Implement quantum feature engineering techniques
    """
    # Generate sample data
    data = generate_credit_data(100)
    
    # Classical feature engineering
    classical_features = data.copy()
    classical_features['debt_income_ratio'] = data['debt'] / (data['income'] + 1)
    classical_features['credit_utilization'] = data['credit_used'] / (data['credit_limit'] + 1)
    
    # Quantum feature engineering using quantum circuits
    quantum_features = quantum_feature_engineering(data)
    
    print("Classical Features Shape:", classical_features.shape)
    print("Quantum Features Shape:", quantum_features.shape)
    
    return classical_features, quantum_features

def quantum_feature_engineering(data):
    """
    Implement quantum feature engineering
    """
    # Create quantum circuit for feature transformation
    num_qubits = 4
    circuit = QuantumCircuit(num_qubits)
    
    # Encode data into quantum state
    for i, row in data.iterrows():
        # Normalize features
        normalized_features = normalize_features(row)
        
        # Apply quantum transformations
        for j, feature in enumerate(normalized_features[:num_qubits]):
            circuit.rx(feature * np.pi, j)
        
        # Add entanglement
        for j in range(num_qubits - 1):
            circuit.cx(j, j + 1)
    
    # Measure quantum features
    circuit.measure_all()
    
    # Execute circuit
    backend = Aer.get_backend('qasm_simulator')
    job = execute(circuit, backend, shots=1000)
    result = job.result()
    counts = result.get_counts()
    
    # Convert quantum measurements to features
    quantum_features = extract_quantum_features(counts, len(data))
    
    return quantum_features

def normalize_features(row):
    """
    Normalize features to [0, 1] range
    """
    features = []
    for col in row.index:
        if col != 'default':
            # Simple normalization
            normalized = (row[col] - row[col].min()) / (row[col].max() - row[col].min())
            features.append(normalized)
    return features

def extract_quantum_features(counts, n_samples):
    """
    Extract quantum features from measurement counts
    """
    # Convert quantum measurements to feature matrix
    feature_matrix = np.zeros((n_samples, 16))  # 2^4 = 16 possible states
    
    for i, (state, count) in enumerate(counts.items()):
        if i < n_samples:
            # Convert binary state to index
            state_index = int(state, 2)
            feature_matrix[i, state_index] = count / 1000  # Normalize by shots
    
    return feature_matrix

# Run the comparison
if __name__ == "__main__":
    print("Comparing Classical vs Quantum Credit Scoring Models...")
    results = compare_models()
    
    print("\nQuantum Feature Engineering Exercise:")
    classical_features, quantum_features = quantum_feature_engineering_exercise()
```

### **Exercise 2: Quantum Credit Score Interpretation**

```python
def quantum_credit_score_interpretation():
    """
    Exercise: Interpret quantum credit scores và explain decisions
    """
    # Generate sample customer data
    customer_data = pd.DataFrame({
        'income': [60000],
        'debt': [30000],
        'credit_used': [15000],
        'credit_limit': [30000],
        'payments_made': [10],
        'payments_due': [12],
        'age': [35],
        'employment_years': [8]
    })
    
    # Calculate classical credit score
    classical_score = calculate_classical_score(customer_data)
    
    # Calculate quantum credit score
    quantum_score = calculate_quantum_score(customer_data)
    
    # Explain the differences
    explain_score_differences(classical_score, quantum_score, customer_data)
    
    return classical_score, quantum_score

def calculate_classical_score(data):
    """
    Calculate classical credit score
    """
    # Simple scoring model
    income_score = min(data['income'].iloc[0] / 10000, 100)
    debt_ratio = data['debt'].iloc[0] / data['income'].iloc[0]
    debt_score = max(100 - debt_ratio * 100, 0)
    payment_score = (data['payments_made'].iloc[0] / data['payments_due'].iloc[0]) * 100
    
    classical_score = (income_score + debt_score + payment_score) / 3
    return classical_score

def calculate_quantum_score(data):
    """
    Calculate quantum credit score
    """
    # Quantum scoring using superposition of multiple factors
    quantum_circuit = create_quantum_scoring_circuit(data)
    
    # Execute quantum circuit
    backend = Aer.get_backend('qasm_simulator')
    job = execute(quantum_circuit, backend, shots=1000)
    result = job.result()
    counts = result.get_counts()
    
    # Calculate quantum score from measurements
    quantum_score = extract_quantum_score(counts)
    return quantum_score

def create_quantum_scoring_circuit(data):
    """
    Create quantum circuit for credit scoring
    """
    num_qubits = 4
    circuit = QuantumCircuit(num_qubits, num_qubits)
    
    # Encode customer data into quantum state
    row = data.iloc[0]
    
    # Normalize and encode features
    income_norm = min(row['income'] / 100000, 1.0)
    debt_ratio = min(row['debt'] / row['income'], 1.0)
    credit_util = min(row['credit_used'] / row['credit_limit'], 1.0)
    payment_ratio = min(row['payments_made'] / row['payments_due'], 1.0)
    
    # Apply quantum transformations
    circuit.rx(income_norm * np.pi, 0)
    circuit.ry(debt_ratio * np.pi, 1)
    circuit.rz(credit_util * np.pi, 2)
    circuit.rx(payment_ratio * np.pi, 3)
    
    # Add entanglement for feature interactions
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(2, 3)
    circuit.cx(3, 0)
    
    # Measure
    circuit.measure_all()
    
    return circuit

def extract_quantum_score(counts):
    """
    Extract quantum score from measurement counts
    """
    total_shots = sum(counts.values())
    score = 0
    
    for state, count in counts.items():
        # Convert binary state to score
        state_value = int(state, 2)
        normalized_value = state_value / 15  # Normalize to [0, 1]
        score += (normalized_value * count) / total_shots
    
    return score * 100  # Scale to [0, 100]

def explain_score_differences(classical_score, quantum_score, data):
    """
    Explain differences between classical and quantum scores
    """
    print(f"Customer Data:")
    print(data.to_string())
    print(f"\nClassical Credit Score: {classical_score:.2f}")
    print(f"Quantum Credit Score: {quantum_score:.2f}")
    print(f"Difference: {abs(classical_score - quantum_score):.2f}")
    
    if abs(classical_score - quantum_score) > 10:
        print("\nSignificant difference detected!")
        print("Quantum model captures additional feature interactions")
        print("Classical model uses linear assumptions")
    
    return classical_score, quantum_score

# Run interpretation exercise
if __name__ == "__main__":
    classical_score, quantum_score = quantum_credit_score_interpretation()
```

## 📊 Kết quả và Phân tích

### **Performance Comparison:**

#### **Classical Credit Scoring:**
- **Logistic Regression**: Linear, interpretable, fast
- **Random Forest**: Non-linear, feature importance
- **Gradient Boosting**: High performance, complex
- **Neural Networks**: Deep learning, black-box

#### **Quantum Credit Scoring:**
- **Quantum Feature Maps**: High-dimensional encoding
- **Quantum Neural Networks**: Superposition-based learning
- **Quantum SVM**: Quantum kernel methods
- **Quantum Clustering**: Quantum pattern recognition

### **Key Insights:**

#### **1. Quantum Advantages:**
- **Feature Interactions**: Entanglement captures complex relationships
- **High-dimensional Data**: Quantum feature maps handle many features
- **Non-linear Patterns**: Quantum transformations capture non-linear effects
- **Uncertainty Quantification**: Quantum measurements provide uncertainty

#### **2. Classical Advantages:**
- **Interpretability**: Clear model explanations
- **Speed**: Fast training và prediction
- **Maturity**: Well-established methods
- **Regulatory Compliance**: Accepted by regulators

#### **3. Hybrid Approach:**
- Use quantum cho complex feature interactions
- Use classical cho interpretable results
- Combine both cho optimal performance

## 🎯 Bài tập về nhà

### **Exercise 1: Quantum Credit Score Calibration**
Implement quantum calibration methods cho credit scores.

### **Exercise 2: Quantum Fairness in Credit Scoring**
Build quantum models ensuring fairness và avoiding bias.

### **Exercise 3: Quantum Real-time Credit Scoring**
Create quantum real-time credit scoring system.

### **Exercise 4: Quantum Credit Score Validation**
Implement validation framework cho quantum credit scoring.

---

> *"Quantum machine learning enables more sophisticated credit scoring by capturing complex, non-linear relationships in high-dimensional feature spaces."* - Quantum Finance Research

> Ngày tiếp theo: [Portfolio Credit Risk](Day4.md) 