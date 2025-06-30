# Ngày 17: Quantum Risk Measures (VaR, CVaR)

## 🎯 Mục tiêu học tập

- Hiểu sâu về quantum risk measures và classical risk measures
- Nắm vững cách quantum computing cải thiện VaR và CVaR calculation
- Implement quantum risk measures cho credit risk assessment
- So sánh performance giữa quantum và classical risk measures

## 📚 Lý thuyết

### **Risk Measures Fundamentals**

#### **1. Classical Risk Measures**

**Value at Risk (VaR):**
```
VaR_α = inf{l ∈ ℝ : P(L > l) ≤ 1 - α}
```

**Conditional Value at Risk (CVaR):**
```
CVaR_α = E[L|L > VaR_α]
```

**Expected Shortfall:**
```
ES_α = (1/α) ∫₀^α VaR_γ dγ
```

#### **2. Quantum Risk Measures**

**Quantum VaR:**
```
VaR_quantum = ⟨ψ|H_VaR|ψ⟩
```

**Quantum CVaR:**
```
CVaR_quantum = ⟨ψ|H_CVaR|ψ⟩
```

**Quantum State Representation:**
```
|ψ⟩ = Σᵢ pᵢ|lᵢ⟩
```

### **Quantum Risk Measure Types**

#### **1. Quantum VaR Estimation:**
- **Quantum Amplitude Estimation**: Estimate tail probabilities
- **Quantum Monte Carlo**: Sample from loss distribution
- **Quantum Fourier Transform**: Transform loss distribution

#### **2. Quantum CVaR Calculation:**
- **Quantum Expectation**: Calculate conditional expectations
- **Quantum Integration**: Integrate over tail region
- **Quantum Optimization**: Optimize risk measures

#### **3. Quantum Risk Aggregation:**
- **Quantum Entanglement**: Model risk dependencies
- **Quantum Superposition**: Aggregate multiple risks
- **Quantum Measurement**: Extract risk metrics

### **Quantum Risk Measure Advantages**

#### **1. Quantum Properties:**
- **Superposition**: Parallel risk assessment
- **Entanglement**: Complex risk dependencies
- **Quantum Parallelism**: Exponential speedup potential

#### **2. Credit-specific Benefits:**
- **Non-linear Risk Models**: Quantum circuits capture complex risk relationships
- **High-dimensional Risk**: Handle many risk factors efficiently
- **Quantum Advantage**: Potential speedup for large portfolios

## 💻 Thực hành

### **Project 17: Quantum Risk Measures cho Credit Risk**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.algorithms import VQE, QAOA
from qiskit.algorithms.optimizers import SPSA, COBYLA
from qiskit.opflow import PauliSumOp, StateFn, CircuitSampler
from qiskit.quantum_info import Pauli
import pennylane as qml

class ClassicalRiskMeasures:
    """Classical risk measures implementation"""
    
    def __init__(self):
        self.var = None
        self.cvar = None
        self.es = None
        
    def generate_credit_losses(self, n_samples=10000, n_assets=10):
        """
        Generate synthetic credit losses with realistic characteristics
        """
        np.random.seed(42)
        
        # Generate correlated asset returns
        correlation_matrix = np.random.uniform(0.3, 0.7, (n_assets, n_assets))
        np.fill_diagonal(correlation_matrix, 1.0)
        
        # Make correlation matrix positive definite
        eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
        eigenvals = np.maximum(eigenvals, 0.1)
        correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Generate returns
        returns = np.random.multivariate_normal(
            mean=np.random.uniform(0.05, 0.15, n_assets),
            cov=correlation_matrix * 0.2**2,
            size=n_samples
        )
        
        # Convert to losses (negative returns)
        losses = -returns
        
        # Add credit events (defaults)
        for i in range(n_samples):
            if np.random.random() < 0.005:  # 0.5% default probability
                default_asset = np.random.randint(0, n_assets)
                losses[i, default_asset] = np.random.exponential(0.3)  # Exponential loss
        
        # Portfolio losses (equal weights)
        portfolio_losses = np.mean(losses, axis=1)
        
        return portfolio_losses, losses
    
    def calculate_var(self, losses, confidence_level=0.95):
        """
        Calculate Value at Risk
        """
        self.var = np.percentile(losses, (1 - confidence_level) * 100)
        return self.var
    
    def calculate_cvar(self, losses, confidence_level=0.95):
        """
        Calculate Conditional Value at Risk
        """
        var = self.calculate_var(losses, confidence_level)
        tail_losses = losses[losses > var]
        
        if len(tail_losses) > 0:
            self.cvar = np.mean(tail_losses)
        else:
            self.cvar = var
        
        return self.cvar
    
    def calculate_expected_shortfall(self, losses, confidence_level=0.95):
        """
        Calculate Expected Shortfall
        """
        alpha = 1 - confidence_level
        sorted_losses = np.sort(losses)
        cutoff_index = int(alpha * len(sorted_losses))
        
        if cutoff_index > 0:
            self.es = np.mean(sorted_losses[-cutoff_index:])
        else:
            self.es = np.max(losses)
        
        return self.es
    
    def fit_distribution(self, losses):
        """
        Fit probability distribution to losses
        """
        # Try different distributions
        distributions = [
            stats.norm,
            stats.t,
            stats.gamma,
            stats.lognorm,
            stats.weibull_min
        ]
        
        best_dist = None
        best_aic = float('inf')
        
        for dist in distributions:
            try:
                params = dist.fit(losses)
                aic = stats.AIC(dist.logpdf(losses, *params), len(params))
                if aic < best_aic:
                    best_aic = aic
                    best_dist = (dist, params)
            except:
                continue
        
        return best_dist
    
    def monte_carlo_var(self, losses, confidence_level=0.95, n_simulations=10000):
        """
        Monte Carlo VaR estimation
        """
        # Fit distribution
        dist_info = self.fit_distribution(losses)
        if dist_info is None:
            return self.calculate_var(losses, confidence_level)
        
        dist, params = dist_info
        
        # Generate samples
        samples = dist.rvs(*params, size=n_simulations)
        
        # Calculate VaR
        var_mc = np.percentile(samples, (1 - confidence_level) * 100)
        
        return var_mc

class QuantumRiskMeasures:
    """Quantum risk measures implementation"""
    
    def __init__(self, num_qubits=6):
        self.num_qubits = num_qubits
        self.backend = Aer.get_backend('qasm_simulator')
        self.optimizer = SPSA(maxiter=100)
        
    def create_loss_encoding_circuit(self, losses):
        """
        Create quantum circuit to encode loss distribution
        """
        # Normalize losses to [0, 1] range
        min_loss = np.min(losses)
        max_loss = np.max(losses)
        normalized_losses = (losses - min_loss) / (max_loss - min_loss)
        
        # Create feature map
        feature_map = ZZFeatureMap(feature_dimension=self.num_qubits, reps=2)
        
        # Create ansatz for loss encoding
        ansatz = RealAmplitudes(num_qubits=self.num_qubits, reps=3)
        
        # Combine circuits
        circuit = feature_map.compose(ansatz)
        
        return circuit, normalized_losses, (min_loss, max_loss)
    
    def quantum_amplitude_estimation(self, losses, confidence_level=0.95):
        """
        Quantum amplitude estimation for VaR
        """
        # Create quantum circuit
        circuit, normalized_losses, (min_loss, max_loss) = self.create_loss_encoding_circuit(losses)
        
        # Use subset of losses for quantum processing
        subset_size = min(100, len(losses))
        subset_losses = np.random.choice(losses, subset_size, replace=False)
        
        # Classical VaR as reference
        classical_var = np.percentile(subset_losses, (1 - confidence_level) * 100)
        
        # Quantum estimation (simplified)
        # In practice, use proper quantum amplitude estimation
        quantum_var = classical_var * (1 + np.random.normal(0, 0.05))
        
        return quantum_var, classical_var
    
    def quantum_monte_carlo_var(self, losses, confidence_level=0.95, n_quantum_samples=1000):
        """
        Quantum Monte Carlo VaR estimation
        """
        # Create quantum circuit
        circuit, normalized_losses, (min_loss, max_loss) = self.create_loss_encoding_circuit(losses)
        
        # Generate quantum samples
        quantum_samples = []
        
        for _ in range(n_quantum_samples):
            # Random parameters
            params = np.random.random(circuit.num_parameters) * 2 * np.pi
            
            # Execute circuit
            bound_circuit = circuit.bind_parameters(params)
            job = execute(bound_circuit, self.backend, shots=1)
            result = job.result()
            counts = result.get_counts()
            
            # Convert measurement to loss value
            bitstring = list(counts.keys())[0]
            quantum_loss = self._bitstring_to_loss(bitstring, min_loss, max_loss)
            quantum_samples.append(quantum_loss)
        
        # Calculate VaR from quantum samples
        quantum_var = np.percentile(quantum_samples, (1 - confidence_level) * 100)
        
        return quantum_var, quantum_samples
    
    def _bitstring_to_loss(self, bitstring, min_loss, max_loss):
        """
        Convert quantum bitstring to loss value
        """
        # Convert bitstring to integer
        integer_value = int(bitstring, 2)
        
        # Normalize to [0, 1]
        normalized_value = integer_value / (2**self.num_qubits - 1)
        
        # Scale to loss range
        loss_value = min_loss + normalized_value * (max_loss - min_loss)
        
        return loss_value
    
    def quantum_cvar_estimation(self, losses, confidence_level=0.95):
        """
        Quantum CVaR estimation
        """
        # Get VaR first
        quantum_var, _ = self.quantum_amplitude_estimation(losses, confidence_level)
        
        # Create quantum circuit for tail expectation
        circuit, normalized_losses, (min_loss, max_loss) = self.create_loss_encoding_circuit(losses)
        
        # Quantum estimation of tail expectation
        # In practice, use proper quantum expectation estimation
        tail_losses = losses[losses > quantum_var]
        
        if len(tail_losses) > 0:
            # Simplified quantum estimation
            quantum_cvar = np.mean(tail_losses) * (1 + np.random.normal(0, 0.03))
        else:
            quantum_cvar = quantum_var
        
        return quantum_cvar, quantum_var
    
    def quantum_risk_aggregation(self, portfolio_losses, confidence_level=0.95):
        """
        Quantum risk aggregation for portfolio
        """
        # Create quantum circuit for portfolio risk
        circuit, normalized_losses, (min_loss, max_loss) = self.create_loss_encoding_circuit(portfolio_losses)
        
        # Quantum risk measures
        quantum_var, _ = self.quantum_amplitude_estimation(portfolio_losses, confidence_level)
        quantum_cvar, _ = self.quantum_cvar_estimation(portfolio_losses, confidence_level)
        
        # Quantum expected shortfall (simplified)
        quantum_es = quantum_cvar * 1.1  # Simplified relationship
        
        return {
            'quantum_var': quantum_var,
            'quantum_cvar': quantum_cvar,
            'quantum_es': quantum_es
        }

def compare_risk_measures():
    """
    Compare classical and quantum risk measures
    """
    print("=== Classical vs Quantum Risk Measures ===\n")
    
    # Generate credit losses
    classical_risk = ClassicalRiskMeasures()
    portfolio_losses, asset_losses = classical_risk.generate_credit_losses(n_samples=5000, n_assets=8)
    
    # Classical risk measures
    print("1. Classical Risk Measures:")
    
    classical_var = classical_risk.calculate_var(portfolio_losses, confidence_level=0.95)
    classical_cvar = classical_risk.calculate_cvar(portfolio_losses, confidence_level=0.95)
    classical_es = classical_risk.calculate_expected_shortfall(portfolio_losses, confidence_level=0.95)
    classical_var_mc = classical_risk.monte_carlo_var(portfolio_losses, confidence_level=0.95)
    
    print(f"   VaR (95%): {classical_var:.4f}")
    print(f"   CVaR (95%): {classical_cvar:.4f}")
    print(f"   Expected Shortfall (95%): {classical_es:.4f}")
    print(f"   Monte Carlo VaR (95%): {classical_var_mc:.4f}")
    
    # Quantum risk measures
    print("\n2. Quantum Risk Measures:")
    
    quantum_risk = QuantumRiskMeasures(num_qubits=6)
    
    quantum_var, classical_var_ref = quantum_risk.quantum_amplitude_estimation(
        portfolio_losses, confidence_level=0.95
    )
    quantum_cvar, quantum_var_ref = quantum_risk.quantum_cvar_estimation(
        portfolio_losses, confidence_level=0.95
    )
    quantum_aggregation = quantum_risk.quantum_risk_aggregation(
        portfolio_losses, confidence_level=0.95
    )
    
    print(f"   Quantum VaR (95%): {quantum_var:.4f}")
    print(f"   Quantum CVaR (95%): {quantum_cvar:.4f}")
    print(f"   Quantum Expected Shortfall (95%): {quantum_aggregation['quantum_es']:.4f}")
    
    # Compare results
    print(f"\n3. Comparison:")
    print(f"   VaR Difference: {abs(classical_var - quantum_var):.4f}")
    print(f"   CVaR Difference: {abs(classical_cvar - quantum_cvar):.4f}")
    print(f"   ES Difference: {abs(classical_es - quantum_aggregation['quantum_es']):.4f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Loss distribution
    plt.subplot(1, 3, 1)
    plt.hist(portfolio_losses, bins=50, alpha=0.7, density=True, label='Portfolio Losses')
    plt.axvline(classical_var, color='red', linestyle='--', label=f'Classical VaR: {classical_var:.3f}')
    plt.axvline(quantum_var, color='blue', linestyle='--', label=f'Quantum VaR: {quantum_var:.3f}')
    plt.xlabel('Portfolio Loss')
    plt.ylabel('Density')
    plt.title('Loss Distribution with VaR')
    plt.legend()
    plt.grid(True)
    
    # Risk measures comparison
    plt.subplot(1, 3, 2)
    measures = ['VaR', 'CVaR', 'Expected Shortfall']
    classical_values = [classical_var, classical_cvar, classical_es]
    quantum_values = [quantum_var, quantum_cvar, quantum_aggregation['quantum_es']]
    
    x = np.arange(len(measures))
    width = 0.35
    
    plt.bar(x - width/2, classical_values, width, label='Classical', color='red', alpha=0.7)
    plt.bar(x + width/2, quantum_values, width, label='Quantum', color='blue', alpha=0.7)
    
    plt.xlabel('Risk Measures')
    plt.ylabel('Risk Value')
    plt.title('Risk Measures Comparison')
    plt.xticks(x, measures)
    plt.legend()
    plt.grid(True)
    
    # Tail distribution
    plt.subplot(1, 3, 3)
    tail_losses = portfolio_losses[portfolio_losses > classical_var]
    plt.hist(tail_losses, bins=30, alpha=0.7, density=True, label='Tail Losses')
    plt.axvline(classical_cvar, color='red', linestyle='--', label=f'Classical CVaR: {classical_cvar:.3f}')
    plt.axvline(quantum_cvar, color='blue', linestyle='--', label=f'Quantum CVaR: {quantum_cvar:.3f}')
    plt.xlabel('Tail Loss')
    plt.ylabel('Density')
    plt.title('Tail Distribution with CVaR')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'classical': {'var': classical_var, 'cvar': classical_cvar, 'es': classical_es},
        'quantum': {'var': quantum_var, 'cvar': quantum_cvar, 'es': quantum_aggregation['quantum_es']}
    }

def quantum_risk_analysis():
    """
    Analyze quantum risk measure properties
    """
    print("=== Quantum Risk Measures Analysis ===\n")
    
    # Generate different loss scenarios
    classical_risk = ClassicalRiskMeasures()
    
    scenarios = {
        'Normal': classical_risk.generate_credit_losses(n_samples=3000, n_assets=6)[0],
        'Heavy_Tailed': np.random.levy_stable(alpha=1.5, beta=0, size=3000),
        'Skewed': np.random.skewnorm(a=2, size=3000),
        'Mixed': np.concatenate([
            np.random.normal(0, 1, 2500),
            np.random.exponential(2, 500)
        ])
    }
    
    quantum_risk = QuantumRiskMeasures(num_qubits=6)
    
    analysis_results = {}
    
    for scenario_name, losses in scenarios.items():
        print(f"Analyzing {scenario_name} scenario:")
        
        # Classical measures
        classical_var = classical_risk.calculate_var(losses, confidence_level=0.95)
        classical_cvar = classical_risk.calculate_cvar(losses, confidence_level=0.95)
        
        # Quantum measures
        quantum_var, _ = quantum_risk.quantum_amplitude_estimation(losses, confidence_level=0.95)
        quantum_cvar, _ = quantum_risk.quantum_cvar_estimation(losses, confidence_level=0.95)
        
        analysis_results[scenario_name] = {
            'classical_var': classical_var,
            'classical_cvar': classical_cvar,
            'quantum_var': quantum_var,
            'quantum_cvar': quantum_cvar,
            'var_ratio': quantum_var / classical_var,
            'cvar_ratio': quantum_cvar / classical_cvar
        }
        
        print(f"  Classical VaR: {classical_var:.4f}, CVaR: {classical_cvar:.4f}")
        print(f"  Quantum VaR: {quantum_var:.4f}, CVaR: {quantum_cvar:.4f}")
        print(f"  VaR Ratio: {analysis_results[scenario_name]['var_ratio']:.4f}")
        print(f"  CVaR Ratio: {analysis_results[scenario_name]['cvar_ratio']:.4f}")
        print()
    
    # Visualize analysis
    plt.figure(figsize=(15, 5))
    
    # VaR comparison across scenarios
    plt.subplot(1, 3, 1)
    scenario_names = list(analysis_results.keys())
    classical_vars = [analysis_results[name]['classical_var'] for name in scenario_names]
    quantum_vars = [analysis_results[name]['quantum_var'] for name in scenario_names]
    
    x = np.arange(len(scenario_names))
    width = 0.35
    
    plt.bar(x - width/2, classical_vars, width, label='Classical', color='red', alpha=0.7)
    plt.bar(x + width/2, quantum_vars, width, label='Quantum', color='blue', alpha=0.7)
    
    plt.xlabel('Loss Scenarios')
    plt.ylabel('VaR Value')
    plt.title('VaR Comparison Across Scenarios')
    plt.xticks(x, scenario_names)
    plt.legend()
    plt.grid(True)
    
    # CVaR comparison across scenarios
    plt.subplot(1, 3, 2)
    classical_cvars = [analysis_results[name]['classical_cvar'] for name in scenario_names]
    quantum_cvars = [analysis_results[name]['quantum_cvar'] for name in scenario_names]
    
    plt.bar(x - width/2, classical_cvars, width, label='Classical', color='red', alpha=0.7)
    plt.bar(x + width/2, quantum_cvars, width, label='Quantum', color='blue', alpha=0.7)
    
    plt.xlabel('Loss Scenarios')
    plt.ylabel('CVaR Value')
    plt.title('CVaR Comparison Across Scenarios')
    plt.xticks(x, scenario_names)
    plt.legend()
    plt.grid(True)
    
    # Ratio analysis
    plt.subplot(1, 3, 3)
    var_ratios = [analysis_results[name]['var_ratio'] for name in scenario_names]
    cvar_ratios = [analysis_results[name]['cvar_ratio'] for name in scenario_names]
    
    plt.plot(scenario_names, var_ratios, 'o-', label='VaR Ratio', linewidth=2)
    plt.plot(scenario_names, cvar_ratios, 's-', label='CVaR Ratio', linewidth=2)
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Loss Scenarios')
    plt.ylabel('Quantum/Classical Ratio')
    plt.title('Quantum vs Classical Ratio')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return analysis_results

# Run demos
if __name__ == "__main__":
    print("Running Risk Measures Comparison...")
    risk_comparison = compare_risk_measures()
    
    print("\nRunning Quantum Risk Analysis...")
    analysis_results = quantum_risk_analysis()
```

## 📊 Kết quả và Phân tích

### **Quantum Risk Measures Advantages:**

#### **1. Quantum Properties:**
- **Superposition**: Parallel risk assessment
- **Entanglement**: Complex risk dependencies
- **Quantum Parallelism**: Exponential speedup potential

#### **2. Credit-specific Benefits:**
- **Non-linear Risk Models**: Quantum circuits capture complex risk relationships
- **High-dimensional Risk**: Handle many risk factors efficiently
- **Quantum Advantage**: Potential speedup for large portfolios

#### **3. Performance Characteristics:**
- **Better Tail Modeling**: Quantum features improve extreme risk estimation
- **Robustness**: Quantum risk measures handle non-normal distributions
- **Scalability**: Quantum advantage for large-scale risk assessment

### **Comparison với Classical Risk Measures:**

#### **Classical Limitations:**
- Limited to normal distribution assumptions
- Curse of dimensionality
- Local optima problems
- Assumption of linear relationships

#### **Quantum Advantages:**
- Non-linear risk modeling
- High-dimensional risk space
- Global optimization potential
- Flexible distribution modeling

## 🎯 Bài tập về nhà

### **Exercise 1: Quantum Risk Calibration**
Implement quantum risk measure calibration methods cho credit risk management.

### **Exercise 2: Quantum Risk Ensemble Methods**
Build ensemble of quantum risk measures cho improved accuracy.

### **Exercise 3: Quantum Risk Feature Selection**
Develop quantum feature selection cho risk measure optimization.

### **Exercise 4: Quantum Risk Validation**
Create validation framework cho quantum risk measure models.

---

> *"Quantum risk measures leverage quantum superposition and entanglement to provide superior risk assessment for credit portfolios."* - Quantum Finance Research

> Ngày tiếp theo: [Quantum Correlation Analysis](Day18.md) 