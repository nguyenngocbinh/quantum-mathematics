# Ngày 2: Mô hình xác suất vỡ nợ truyền thống

## 🎯 Mục tiêu học tập

- Hiểu sâu về các mô hình xác suất vỡ nợ truyền thống
- Phân tích hạn chế của classical models
- Implement quantum-enhanced default probability models
- So sánh performance giữa classical và quantum approaches

## 📚 Lý thuyết

### **Mô hình xác suất vỡ nợ truyền thống**

#### **1. Merton Model (1974)**
Mô hình đầu tiên sử dụng option pricing theory cho credit risk:

**Công thức:**
```
PD = N(-d2)
d2 = [ln(V/F) + (r - σ²/2)T] / (σ√T)
```

Trong đó:
- PD: Probability of Default
- V: Asset value
- F: Face value of debt
- r: Risk-free rate
- σ: Asset volatility
- T: Time to maturity

#### **2. KMV Model**
Extension của Merton model với empirical adjustments:

**Distance to Default:**
```
DD = (V - F) / (σV)
```

#### **3. CreditMetrics (JP Morgan)**
Portfolio-based approach sử dụng transition matrices:

**Expected Loss:**
```
EL = PD × LGD × EAD
```

#### **4. Credit Risk+ (Credit Suisse)**
Actuarial approach với Poisson distribution:

**Portfolio Loss Distribution:**
```
P(L = k) = Σ P(λ) × Poisson(k|λ)
```

### **Hạn chế của Classical Models**

#### **1. Distributional Assumptions:**
- Gaussian assumptions không phù hợp với fat tails
- Linear correlations breakdown trong stress scenarios
- Stationarity assumptions trong non-stationary markets

#### **2. Computational Limitations:**
- Monte Carlo simulation cho large portfolios
- Numerical integration cho complex payoffs
- Real-time risk assessment challenges

#### **3. Data Challenges:**
- Sparse default data
- Missing market data
- Model calibration issues

### **Quantum Advantages cho Default Modeling**

#### **1. Quantum Monte Carlo:**
- Exponential speedup cho certain simulations
- True quantum randomness
- Parallel scenario processing

#### **2. Quantum Feature Maps:**
- High-dimensional feature encoding
- Non-linear relationship modeling
- Quantum kernel methods

#### **3. Quantum Optimization:**
- Portfolio optimization với quantum algorithms
- Risk measure optimization
- Model calibration

## 💻 Thực hành

### **Project 2: Quantum-Enhanced Default Probability Model**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector
from qiskit.algorithms import VQE, QAOA
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import TwoLocal
import pennylane as qml

class ClassicalDefaultModel:
    """Classical Merton model implementation"""
    
    def __init__(self):
        self.risk_free_rate = 0.02
        
    def merton_pd(self, asset_value, debt_value, volatility, time_to_maturity):
        """
        Calculate default probability using Merton model
        """
        # Calculate d2 parameter
        d2 = (np.log(asset_value / debt_value) + 
              (self.risk_free_rate - 0.5 * volatility**2) * time_to_maturity) / \
             (volatility * np.sqrt(time_to_maturity))
        
        # Default probability
        pd = norm.cdf(-d2)
        return pd
    
    def distance_to_default(self, asset_value, debt_value, volatility, time_to_maturity):
        """
        Calculate distance to default
        """
        dd = (np.log(asset_value / debt_value) + 
              (self.risk_free_rate - 0.5 * volatility**2) * time_to_maturity) / \
             (volatility * np.sqrt(time_to_maturity))
        return dd

class QuantumDefaultModel:
    """Quantum-enhanced default probability model"""
    
    def __init__(self, num_qubits=4):
        self.num_qubits = num_qubits
        self.backend = Aer.get_backend('qasm_simulator')
        
    def create_quantum_circuit(self, asset_value, debt_value, volatility, time_to_maturity):
        """
        Create quantum circuit for default probability calculation
        """
        circuit = QuantumCircuit(self.num_qubits, self.num_qubits)
        
        # Encode financial parameters into quantum state
        # Normalize parameters to [0, 1]
        asset_ratio = min(asset_value / debt_value, 3.0) / 3.0
        vol_norm = min(volatility, 1.0)
        time_norm = min(time_to_maturity, 10.0) / 10.0
        
        # Apply rotations based on parameters
        circuit.rx(asset_ratio * np.pi, 0)
        circuit.ry(vol_norm * np.pi, 1)
        circuit.rz(time_norm * np.pi, 2)
        
        # Add entanglement
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.cx(2, 3)
        
        # Add measurement
        circuit.measure_all()
        
        return circuit
    
    def calculate_quantum_pd(self, asset_value, debt_value, volatility, time_to_maturity, shots=1000):
        """
        Calculate default probability using quantum circuit
        """
        circuit = self.create_quantum_circuit(asset_value, debt_value, volatility, time_to_maturity)
        
        # Execute circuit
        job = execute(circuit, self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        # Calculate default probability from measurement results
        default_prob = self._extract_pd_from_counts(counts)
        return default_prob
    
    def _extract_pd_from_counts(self, counts):
        """
        Extract default probability from measurement counts
        """
        total_shots = sum(counts.values())
        default_count = 0
        
        for state, count in counts.items():
            # Consider states with specific patterns as default indicators
            # For simplicity, we'll use a heuristic based on bit patterns
            if self._is_default_state(state):
                default_count += count
        
        return default_count / total_shots
    
    def _is_default_state(self, state):
        """
        Determine if a quantum state represents default
        """
        # Simple heuristic: if first two qubits are 1, consider it default
        return state[:2] == '11'

def compare_models():
    """
    Compare classical and quantum default probability models
    """
    # Initialize models
    classical_model = ClassicalDefaultModel()
    quantum_model = QuantumDefaultModel()
    
    # Test parameters
    asset_values = np.linspace(50, 200, 20)
    debt_value = 100
    volatility = 0.3
    time_to_maturity = 1.0
    
    classical_pds = []
    quantum_pds = []
    
    for asset_value in asset_values:
        # Classical calculation
        classical_pd = classical_model.merton_pd(asset_value, debt_value, volatility, time_to_maturity)
        classical_pds.append(classical_pd)
        
        # Quantum calculation
        quantum_pd = quantum_model.calculate_quantum_pd(asset_value, debt_value, volatility, time_to_maturity)
        quantum_pds.append(quantum_pd)
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(asset_values, classical_pds, 'b-', label='Classical Merton', linewidth=2)
    plt.plot(asset_values, quantum_pds, 'r--', label='Quantum Model', linewidth=2)
    plt.xlabel('Asset Value')
    plt.ylabel('Default Probability')
    plt.title('Default Probability Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.scatter(classical_pds, quantum_pds, alpha=0.7)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('Classical PD')
    plt.ylabel('Quantum PD')
    plt.title('Model Correlation')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    differences = np.array(quantum_pds) - np.array(classical_pds)
    plt.plot(asset_values, differences, 'g-', linewidth=2)
    plt.xlabel('Asset Value')
    plt.ylabel('Quantum - Classical PD')
    plt.title('Model Differences')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.hist(differences, bins=10, alpha=0.7, color='orange')
    plt.xlabel('PD Difference')
    plt.ylabel('Frequency')
    plt.title('Distribution of Model Differences')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return asset_values, classical_pds, quantum_pds

# Run comparison
asset_vals, classical_results, quantum_results = compare_models()
```

### **Exercise 2: Portfolio Default Risk Analysis**

```python
def portfolio_default_analysis():
    """
    Analyze default risk for a portfolio of credits
    """
    # Generate synthetic portfolio data
    np.random.seed(42)
    n_credits = 50
    
    # Portfolio parameters
    asset_values = np.random.lognormal(4.5, 0.3, n_credits)  # Log-normal distribution
    debt_values = np.random.uniform(80, 120, n_credits)
    volatilities = np.random.uniform(0.2, 0.5, n_credits)
    time_to_maturity = np.random.uniform(0.5, 3.0, n_credits)
    
    # Create portfolio DataFrame
    portfolio = pd.DataFrame({
        'asset_value': asset_values,
        'debt_value': debt_values,
        'volatility': volatilities,
        'time_to_maturity': time_to_maturity,
        'exposure': debt_values  # Assume exposure equals debt value
    })
    
    # Calculate default probabilities
    classical_model = ClassicalDefaultModel()
    quantum_model = QuantumDefaultModel()
    
    classical_pds = []
    quantum_pds = []
    
    for _, row in portfolio.iterrows():
        classical_pd = classical_model.merton_pd(
            row['asset_value'], row['debt_value'], 
            row['volatility'], row['time_to_maturity']
        )
        classical_pds.append(classical_pd)
        
        quantum_pd = quantum_model.calculate_quantum_pd(
            row['asset_value'], row['debt_value'], 
            row['volatility'], row['time_to_maturity']
        )
        quantum_pds.append(quantum_pd)
    
    portfolio['classical_pd'] = classical_pds
    portfolio['quantum_pd'] = quantum_pds
    
    # Calculate expected losses
    portfolio['classical_el'] = portfolio['classical_pd'] * portfolio['exposure'] * 0.6  # Assume 60% LGD
    portfolio['quantum_el'] = portfolio['quantum_pd'] * portfolio['exposure'] * 0.6
    
    # Portfolio analysis
    print("Portfolio Default Risk Analysis:")
    print(f"Total Portfolio Value: ${portfolio['exposure'].sum():,.0f}")
    print(f"Average Classical PD: {portfolio['classical_pd'].mean():.4f}")
    print(f"Average Quantum PD: {portfolio['quantum_pd'].mean():.4f}")
    print(f"Total Classical EL: ${portfolio['classical_el'].sum():,.0f}")
    print(f"Total Quantum EL: ${portfolio['quantum_el'].sum():,.0f}")
    
    # Risk concentration analysis
    high_risk_classical = portfolio[portfolio['classical_pd'] > 0.1]
    high_risk_quantum = portfolio[portfolio['quantum_pd'] > 0.1]
    
    print(f"\nHigh Risk Credits (Classical): {len(high_risk_classical)}")
    print(f"High Risk Credits (Quantum): {len(high_risk_quantum)}")
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.hist(portfolio['classical_pd'], bins=15, alpha=0.7, label='Classical', color='blue')
    plt.hist(portfolio['quantum_pd'], bins=15, alpha=0.7, label='Quantum', color='red')
    plt.xlabel('Default Probability')
    plt.ylabel('Frequency')
    plt.title('PD Distribution Comparison')
    plt.legend()
    
    plt.subplot(2, 3, 2)
    plt.scatter(portfolio['asset_value'], portfolio['classical_pd'], alpha=0.6, label='Classical')
    plt.scatter(portfolio['asset_value'], portfolio['quantum_pd'], alpha=0.6, label='Quantum')
    plt.xlabel('Asset Value')
    plt.ylabel('Default Probability')
    plt.title('Asset Value vs PD')
    plt.legend()
    
    plt.subplot(2, 3, 3)
    plt.scatter(portfolio['volatility'], portfolio['classical_pd'], alpha=0.6, label='Classical')
    plt.scatter(portfolio['volatility'], portfolio['quantum_pd'], alpha=0.6, label='Quantum')
    plt.xlabel('Volatility')
    plt.ylabel('Default Probability')
    plt.title('Volatility vs PD')
    plt.legend()
    
    plt.subplot(2, 3, 4)
    plt.hist(portfolio['classical_el'], bins=15, alpha=0.7, label='Classical EL', color='blue')
    plt.hist(portfolio['quantum_el'], bins=15, alpha=0.7, label='Quantum EL', color='red')
    plt.xlabel('Expected Loss')
    plt.ylabel('Frequency')
    plt.title('Expected Loss Distribution')
    plt.legend()
    
    plt.subplot(2, 3, 5)
    # Risk-return scatter plot
    plt.scatter(portfolio['classical_pd'], portfolio['exposure'], alpha=0.6, label='Classical')
    plt.scatter(portfolio['quantum_pd'], portfolio['exposure'], alpha=0.6, label='Quantum')
    plt.xlabel('Default Probability')
    plt.ylabel('Exposure')
    plt.title('Risk-Return Profile')
    plt.legend()
    
    plt.subplot(2, 3, 6)
    # Correlation heatmap
    correlation_cols = ['asset_value', 'debt_value', 'volatility', 'time_to_maturity', 
                       'classical_pd', 'quantum_pd']
    correlation_matrix = portfolio[correlation_cols].corr()
    plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(correlation_cols)), correlation_cols, rotation=45)
    plt.yticks(range(len(correlation_cols)), correlation_cols)
    plt.title('Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.show()
    
    return portfolio

# Run portfolio analysis
portfolio_results = portfolio_default_analysis()
```

## 📊 Bài tập về nhà

### **Bài tập 1: Model Calibration**
- Calibrate quantum model parameters cho real market data
- Compare calibration accuracy với classical models
- Analyze parameter sensitivity

### **Bài tập 2: Stress Testing**
- Implement quantum stress testing framework
- Test model performance under extreme scenarios
- Compare với classical stress testing results

### **Bài tập 3: Regulatory Compliance**
- Analyze model compliance với Basel III requirements
- Implement quantum-enhanced regulatory reporting
- Document model validation procedures

## 🔗 Tài liệu tham khảo

### **Papers:**
- "Merton Model and Credit Risk" - Merton (1974)
- "Quantum Monte Carlo Methods in Finance" - Various authors
- "Quantum Computing for Credit Risk Modeling" - Research papers

### **Books:**
- "Credit Risk Modeling" - Lando
- "Quantitative Risk Management" - McNeil, Frey, Embrechts

### **Online Resources:**
- [Qiskit Finance Documentation](https://qiskit.org/ecosystem/finance/)
- [Credit Risk Modeling Tutorials](https://www.risk.net/topics/credit-risk)

## 🎯 Kết luận

Ngày 2 đã cover:
- ✅ Traditional default probability models
- ✅ Limitations của classical approaches
- ✅ Quantum-enhanced default modeling
- ✅ Portfolio-level risk analysis

**Chuẩn bị cho ngày mai**: Credit Scoring và Machine Learning với quantum enhancements.

---

> *"Quantum computing enables us to model complex financial relationships that classical computers cannot efficiently handle."* - Quantum Finance Research 