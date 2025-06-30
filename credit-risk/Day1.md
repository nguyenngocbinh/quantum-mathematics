# Ngày 1: Giới thiệu Credit Risk và Quantum Computing

## 🎯 Mục tiêu học tập

- Hiểu cơ bản về credit risk và các thách thức hiện tại
- Nắm vững cách quantum computing có thể giải quyết các bài toán credit risk
- Thiết lập môi trường phát triển cho quantum finance
- Tạo project đầu tiên: Quantum Credit Risk Assessment

## 📚 Lý thuyết

### **Credit Risk là gì?**

Credit risk là rủi ro mà người vay không thể trả nợ đúng hạn hoặc không trả được nợ. Trong ngành tài chính, đây là một trong những rủi ro quan trọng nhất cần quản lý.

#### **Các loại Credit Risk:**
1. **Default Risk**: Rủi ro vỡ nợ hoàn toàn
2. **Downgrade Risk**: Rủi ro giảm xếp hạng tín dụng
3. **Concentration Risk**: Rủi ro tập trung vào một số đối tượng
4. **Country Risk**: Rủi ro quốc gia

### **Thách thức hiện tại trong Credit Risk Modeling:**

#### **1. Computational Complexity:**
- Portfolio optimization với hàng nghìn assets
- Monte Carlo simulation cho stress testing
- Real-time risk assessment

#### **2. Data Challenges:**
- High-dimensional feature spaces
- Non-linear relationships
- Missing data và data quality issues

#### **3. Model Limitations:**
- Linear assumptions trong non-linear markets
- Gaussian distribution assumptions
- Correlation breakdown trong stress scenarios

### **Quantum Computing Advantage:**

#### **1. Quantum Speedup:**
- **Grover's Algorithm**: Quadratic speedup cho search problems
- **Quantum Monte Carlo**: Exponential speedup cho certain simulations
- **Quantum Machine Learning**: Potential speedup cho training

#### **2. Quantum Features:**
- **Superposition**: Parallel processing of multiple scenarios
- **Entanglement**: Modeling complex correlations
- **Quantum Randomness**: True randomness cho Monte Carlo

#### **3. Specific Applications:**
- **Credit Scoring**: Quantum feature maps cho high-dimensional data
- **Portfolio Optimization**: Quantum optimization algorithms
- **Risk Measures**: Quantum algorithms cho VaR/CVaR calculation

## 💻 Thực hành

### **Setup môi trường:**

```python
# Install required packages
!pip install qiskit-finance
!pip install pennylane
!pip install pandas numpy scipy scikit-learn
!pip install matplotlib seaborn plotly
!pip install yfinance
```

### **Project 1: Quantum Credit Risk Assessment Framework**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_finance.data_providers import RandomDataProvider
import pennylane as qml

class QuantumCreditRiskFramework:
    def __init__(self):
        self.backend = Aer.get_backend('qasm_simulator')
        self.quantum_circuit = None
        
    def create_quantum_credit_state(self, credit_score, risk_factors):
        """
        Tạo quantum state cho credit assessment
        """
        # Number of qubits based on risk factors
        num_qubits = len(risk_factors)
        self.quantum_circuit = QuantumCircuit(num_qubits, num_qubits)
        
        # Encode credit score vào quantum state
        for i, factor in enumerate(risk_factors):
            # Normalize factor to [0, 1]
            normalized_factor = (factor - min(risk_factors)) / (max(risk_factors) - min(risk_factors))
            
            # Apply rotation based on risk factor
            angle = normalized_factor * np.pi
            self.quantum_circuit.rx(angle, i)
            
        # Add entanglement between risk factors
        for i in range(num_qubits - 1):
            self.quantum_circuit.cx(i, i + 1)
            
        return self.quantum_circuit
    
    def measure_credit_risk(self, shots=1000):
        """
        Đo lường credit risk từ quantum state
        """
        if self.quantum_circuit is None:
            raise ValueError("Quantum circuit chưa được tạo")
            
        # Add measurement
        self.quantum_circuit.measure_all()
        
        # Execute circuit
        job = execute(self.quantum_circuit, self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        # Calculate risk probability
        risk_probability = self._calculate_risk_from_counts(counts)
        return risk_probability
    
    def _calculate_risk_from_counts(self, counts):
        """
        Tính toán xác suất rủi ro từ measurement counts
        """
        total_shots = sum(counts.values())
        risk_count = 0
        
        for state, count in counts.items():
            # Consider states with more 1s as higher risk
            ones_count = state.count('1')
            if ones_count > len(state) / 2:
                risk_count += count
                
        return risk_count / total_shots

# Example usage
def demo_quantum_credit_risk():
    """
    Demo quantum credit risk assessment
    """
    # Initialize framework
    qcrf = QuantumCreditRiskFramework()
    
    # Sample credit data
    credit_score = 750
    risk_factors = [0.3, 0.7, 0.5, 0.2, 0.8]  # Various risk factors
    
    # Create quantum state
    circuit = qcrf.create_quantum_credit_state(credit_score, risk_factors)
    
    # Measure risk
    risk_probability = qcrf.measure_credit_risk()
    
    print(f"Credit Score: {credit_score}")
    print(f"Risk Factors: {risk_factors}")
    print(f"Quantum Risk Probability: {risk_probability:.4f}")
    print(f"Risk Level: {'High' if risk_probability > 0.5 else 'Low'}")
    
    return circuit, risk_probability

# Run demo
if __name__ == "__main__":
    circuit, risk = demo_quantum_credit_risk()
    print("\nQuantum Circuit:")
    print(circuit)
```

### **Exercise 1: Quantum Credit Scoring**

```python
def quantum_credit_scoring_exercise():
    """
    Exercise: Implement quantum credit scoring với different datasets
    """
    # Generate synthetic credit data
    np.random.seed(42)
    n_samples = 100
    
    # Simulate credit features
    income = np.random.normal(50000, 20000, n_samples)
    debt_ratio = np.random.uniform(0.1, 0.8, n_samples)
    payment_history = np.random.uniform(0.5, 1.0, n_samples)
    credit_utilization = np.random.uniform(0.1, 0.9, n_samples)
    age = np.random.uniform(25, 65, n_samples)
    
    # Create DataFrame
    credit_data = pd.DataFrame({
        'income': income,
        'debt_ratio': debt_ratio,
        'payment_history': payment_history,
        'credit_utilization': credit_utilization,
        'age': age
    })
    
    # Normalize features
    credit_data_normalized = (credit_data - credit_data.mean()) / credit_data.std()
    
    # Apply quantum credit scoring
    qcrf = QuantumCreditRiskFramework()
    quantum_scores = []
    
    for _, row in credit_data_normalized.iterrows():
        risk_factors = row.values
        qcrf.create_quantum_credit_state(750, risk_factors)
        risk_prob = qcrf.measure_credit_risk()
        quantum_scores.append(1 - risk_prob)  # Convert to credit score
    
    credit_data['quantum_score'] = quantum_scores
    
    # Analyze results
    print("Quantum Credit Scoring Results:")
    print(f"Average Quantum Score: {np.mean(quantum_scores):.4f}")
    print(f"Score Standard Deviation: {np.std(quantum_scores):.4f}")
    print(f"Score Range: [{np.min(quantum_scores):.4f}, {np.max(quantum_scores):.4f}]")
    
    # Visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(quantum_scores, bins=20, alpha=0.7)
    plt.title('Distribution of Quantum Credit Scores')
    plt.xlabel('Quantum Score')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 2, 2)
    plt.scatter(credit_data['income'], quantum_scores, alpha=0.6)
    plt.title('Income vs Quantum Score')
    plt.xlabel('Income')
    plt.ylabel('Quantum Score')
    
    plt.subplot(2, 2, 3)
    plt.scatter(credit_data['debt_ratio'], quantum_scores, alpha=0.6)
    plt.title('Debt Ratio vs Quantum Score')
    plt.xlabel('Debt Ratio')
    plt.ylabel('Quantum Score')
    
    plt.subplot(2, 2, 4)
    correlation_matrix = credit_data[['income', 'debt_ratio', 'payment_history', 
                                     'credit_utilization', 'age', 'quantum_score']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.show()
    
    return credit_data

# Run exercise
credit_results = quantum_credit_scoring_exercise()
```

## 📊 Bài tập về nhà

### **Bài tập 1: Quantum Risk Factor Analysis**
- Tạo quantum circuit cho 3 risk factors khác nhau
- So sánh risk probability với classical methods
- Viết report về quantum advantage

### **Bài tập 2: Credit Portfolio Simulation**
- Simulate portfolio với 10 credit instruments
- Apply quantum optimization cho portfolio weights
- Compare với classical portfolio optimization

### **Bài tập 3: Research Paper Review**
- Đọc paper về "Quantum Computing for Finance"
- Tóm tắt key findings và applications
- Identify potential research gaps

## 🔗 Tài liệu tham khảo

### **Papers:**
- "Quantum Computing for Finance: Overview and Prospects" - Orús et al.
- "Quantum Machine Learning for Credit Risk Assessment" - Various authors

### **Books:**
- "Credit Risk Modeling" - Lando
- "Quantum Computing for Finance" - Orús, Mugel, Lizaso

### **Online Resources:**
- [Qiskit Finance Tutorials](https://qiskit.org/ecosystem/finance/)
- [IBM Quantum Finance](https://quantum-computing.ibm.com/lab/docs/iql/finance/)
- [PennyLane Finance Examples](https://pennylane.ai/qml/demos/)

## 🎯 Kết luận

Ngày 1 đã giới thiệu:
- ✅ Fundamentals của credit risk
- ✅ Quantum computing advantages cho finance
- ✅ Basic quantum credit risk framework
- ✅ Hands-on implementation với Qiskit

**Chuẩn bị cho ngày mai**: Mô hình xác suất vỡ nợ truyền thống và quantum alternatives.

---

> *"Quantum computing offers unprecedented opportunities to solve complex financial problems that are currently intractable with classical methods."* - IBM Research 