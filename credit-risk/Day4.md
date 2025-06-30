# Ngày 4: Portfolio Credit Risk

## 🎯 Mục tiêu học tập

- Hiểu sâu về portfolio credit risk và các mô hình truyền thống
- Phân tích hạn chế của classical portfolio models
- Implement quantum-enhanced portfolio credit risk models
- So sánh performance giữa classical và quantum approaches

## 📚 Lý thuyết

### **Portfolio Credit Risk Fundamentals**

#### **1. Portfolio Credit Risk là gì?**
Portfolio credit risk là rủi ro tổng hợp từ tất cả các khoản vay trong danh mục, bao gồm:
- **Individual Default Risk**: Rủi ro vỡ nợ của từng khoản vay
- **Correlation Risk**: Rủi ro tương quan giữa các khoản vay
- **Concentration Risk**: Rủi ro tập trung vào một số đối tượng
- **Systematic Risk**: Rủi ro hệ thống ảnh hưởng toàn bộ danh mục

#### **2. Key Portfolio Metrics:**
- **Expected Loss (EL)**: Tổn thất kỳ vọng
- **Unexpected Loss (UL)**: Tổn thất ngoài dự kiến
- **Value at Risk (VaR)**: Giá trị rủi ro
- **Conditional VaR (CVaR)**: VaR có điều kiện
- **Economic Capital**: Vốn kinh tế cần thiết

### **Classical Portfolio Credit Risk Models**

#### **1. CreditMetrics (JP Morgan):**
```
EL = Σ (PDᵢ × LGDᵢ × EADᵢ)
UL = √(Σ Σ ρᵢⱼ × ULᵢ × ULⱼ)
```

Trong đó:
- PD: Probability of Default
- LGD: Loss Given Default
- EAD: Exposure at Default
- ρ: Correlation matrix

#### **2. Credit Risk+ (Credit Suisse):**
- Actuarial approach
- Poisson distribution cho defaults
- Analytical solution cho portfolio loss

#### **3. KMV Portfolio Manager:**
- Merton model extension
- Distance to default approach
- Monte Carlo simulation

#### **4. CreditPortfolioView (McKinsey):**
- Macroeconomic factors
- Conditional default probabilities
- Multi-factor model

### **Hạn chế của Classical Models**

#### **1. Correlation Modeling:**
- **Linear Correlations**: Không phù hợp với extreme events
- **Gaussian Assumptions**: Không capture fat tails
- **Static Correlations**: Không thay đổi theo thời gian
- **Pairwise Correlations**: Không capture higher-order dependencies

#### **2. Computational Limitations:**
- **Monte Carlo Complexity**: Exponential với portfolio size
- **Correlation Matrix**: N² parameters cho N assets
- **Real-time Updates**: Không thể cập nhật real-time
- **Large Portfolios**: Computational bottlenecks

#### **3. Model Assumptions:**
- **Stationarity**: Markets không stationary
- **Normality**: Returns không normal
- **Independence**: Defaults không independent
- **Homogeneity**: Assets không homogeneous

### **Quantum Advantages cho Portfolio Credit Risk**

#### **1. Quantum Correlation Modeling:**
- **Entanglement**: Natural modeling of correlations
- **Superposition**: Parallel processing of scenarios
- **Quantum Randomness**: True randomness cho Monte Carlo

#### **2. Quantum Optimization:**
- **Portfolio Optimization**: Quantum algorithms cho optimal weights
- **Risk Minimization**: Quantum algorithms cho risk measures
- **Capital Allocation**: Quantum optimization cho capital

#### **3. Quantum Monte Carlo:**
- **Exponential Speedup**: Cho certain simulations
- **Parallel Scenarios**: Process multiple scenarios simultaneously
- **Quantum Random Walks**: Advanced simulation techniques

## 💻 Thực hành

### **Project 4: Quantum Portfolio Credit Risk Model**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, multivariate_normal
from scipy.optimize import minimize
from qiskit import QuantumCircuit, Aer, execute
from qiskit.algorithms import VQE, QAOA
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import TwoLocal
import pennylane as qml

class ClassicalPortfolioCreditRisk:
    """Classical portfolio credit risk model"""
    
    def __init__(self):
        self.portfolio = None
        self.correlation_matrix = None
        
    def create_portfolio(self, n_assets=10):
        """
        Create synthetic credit portfolio
        """
        np.random.seed(42)
        
        portfolio_data = []
        for i in range(n_assets):
            asset = {
                'asset_id': f'Asset_{i}',
                'exposure': np.random.uniform(100000, 1000000),
                'pd': np.random.uniform(0.01, 0.10),
                'lgd': np.random.uniform(0.3, 0.7),
                'rating': np.random.choice(['AAA', 'AA', 'A', 'BBB', 'BB', 'B']),
                'sector': np.random.choice(['Banking', 'Technology', 'Energy', 'Healthcare', 'Consumer'])
            }
            portfolio_data.append(asset)
        
        self.portfolio = pd.DataFrame(portfolio_data)
        return self.portfolio
    
    def create_correlation_matrix(self, n_assets):
        """
        Create correlation matrix for portfolio
        """
        # Generate random correlation matrix
        np.random.seed(42)
        A = np.random.randn(n_assets, n_assets)
        correlation_matrix = np.dot(A, A.T)
        
        # Normalize to get correlation matrix
        diag_sqrt = np.sqrt(np.diag(correlation_matrix))
        correlation_matrix = correlation_matrix / np.outer(diag_sqrt, diag_sqrt)
        
        # Ensure positive definiteness
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        np.fill_diagonal(correlation_matrix, 1.0)
        
        self.correlation_matrix = correlation_matrix
        return correlation_matrix
    
    def calculate_expected_loss(self):
        """
        Calculate expected loss for portfolio
        """
        if self.portfolio is None:
            raise ValueError("Portfolio chưa được tạo")
        
        expected_loss = np.sum(self.portfolio['exposure'] * 
                              self.portfolio['pd'] * 
                              self.portfolio['lgd'])
        
        return expected_loss
    
    def calculate_unexpected_loss(self):
        """
        Calculate unexpected loss using correlation matrix
        """
        if self.correlation_matrix is None:
            raise ValueError("Correlation matrix chưa được tạo")
        
        # Calculate individual unexpected losses
        individual_uls = self.portfolio['exposure'] * \
                        self.portfolio['pd'] * \
                        self.portfolio['lgd'] * \
                        np.sqrt(self.portfolio['pd'] * (1 - self.portfolio['pd']))
        
        # Calculate portfolio unexpected loss
        portfolio_ul = np.sqrt(np.dot(individual_uls.T, 
                                     np.dot(self.correlation_matrix, individual_uls)))
        
        return portfolio_ul
    
    def monte_carlo_simulation(self, n_simulations=10000):
        """
        Monte Carlo simulation for portfolio loss distribution
        """
        if self.correlation_matrix is None:
            raise ValueError("Correlation matrix chưa được tạo")
        
        n_assets = len(self.portfolio)
        
        # Generate correlated random numbers
        correlated_randoms = multivariate_normal.rvs(
            mean=np.zeros(n_assets),
            cov=self.correlation_matrix,
            size=n_simulations
        )
        
        # Convert to default indicators
        default_thresholds = norm.ppf(self.portfolio['pd'].values)
        defaults = (correlated_randoms < default_thresholds).astype(int)
        
        # Calculate portfolio losses
        portfolio_losses = np.sum(defaults * 
                                 self.portfolio['exposure'].values.reshape(1, -1) * 
                                 self.portfolio['lgd'].values.reshape(1, -1), axis=1)
        
        return portfolio_losses
    
    def calculate_risk_measures(self, portfolio_losses):
        """
        Calculate risk measures from loss distribution
        """
        risk_measures = {
            'expected_loss': np.mean(portfolio_losses),
            'unexpected_loss': np.std(portfolio_losses),
            'var_95': np.percentile(portfolio_losses, 95),
            'var_99': np.percentile(portfolio_losses, 99),
            'cvar_95': np.mean(portfolio_losses[portfolio_losses >= np.percentile(portfolio_losses, 95)]),
            'cvar_99': np.mean(portfolio_losses[portfolio_losses >= np.percentile(portfolio_losses, 99)]),
            'max_loss': np.max(portfolio_losses)
        }
        
        return risk_measures

class QuantumPortfolioCreditRisk:
    """Quantum-enhanced portfolio credit risk model"""
    
    def __init__(self, num_qubits=8):
        self.num_qubits = num_qubits
        self.backend = Aer.get_backend('qasm_simulator')
        self.portfolio = None
        
    def create_quantum_portfolio_circuit(self, portfolio):
        """
        Create quantum circuit for portfolio credit risk
        """
        n_assets = len(portfolio)
        circuit = QuantumCircuit(self.num_qubits, self.num_qubits)
        
        # Encode portfolio information
        for i, (_, asset) in enumerate(portfolio.iterrows()):
            if i < self.num_qubits:
                # Encode exposure
                exposure_norm = min(asset['exposure'] / 1000000, 1.0)
                circuit.rx(exposure_norm * np.pi, i)
                
                # Encode PD
                circuit.ry(asset['pd'] * np.pi, i)
                
                # Encode LGD
                circuit.rz(asset['lgd'] * np.pi, i)
        
        # Add quantum correlations (entanglement)
        for i in range(self.num_qubits - 1):
            circuit.cx(i, i + 1)
        
        # Add additional entanglement for complex correlations
        circuit.cx(0, 2)
        circuit.cx(1, 3)
        
        return circuit
    
    def quantum_monte_carlo_simulation(self, portfolio, n_simulations=1000):
        """
        Quantum Monte Carlo simulation
        """
        circuit = self.create_quantum_portfolio_circuit(portfolio)
        circuit.measure_all()
        
        # Execute quantum simulation
        losses = []
        for sim in range(n_simulations):
            job = execute(circuit, self.backend, shots=1)
            result = job.result()
            counts = result.get_counts()
            
            # Calculate loss from measurement
            loss = self._calculate_loss_from_measurement(counts, portfolio)
            losses.append(loss)
        
        return np.array(losses)
    
    def _calculate_loss_from_measurement(self, counts, portfolio):
        """
        Calculate portfolio loss from quantum measurement
        """
        state = list(counts.keys())[0]
        
        loss = 0
        for i, bit in enumerate(state):
            if i < len(portfolio):
                if bit == '1':  # Default occurred
                    asset = portfolio.iloc[i]
                    loss += asset['exposure'] * asset['lgd']
        
        return loss
    
    def quantum_correlation_analysis(self, portfolio):
        """
        Quantum correlation analysis
        """
        # Create quantum circuit for correlation analysis
        n_assets = len(portfolio)
        circuit = QuantumCircuit(self.num_qubits, self.num_qubits)
        
        # Encode asset characteristics
        for i, (_, asset) in enumerate(portfolio.iterrows()):
            if i < self.num_qubits:
                # Encode sector information
                sector_encoding = hash(asset['sector']) % 4 / 4.0
                circuit.rx(sector_encoding * np.pi, i)
                
                # Encode rating information
                rating_encoding = hash(asset['rating']) % 4 / 4.0
                circuit.ry(rating_encoding * np.pi, i)
        
        # Add entanglement for correlation modeling
        for i in range(self.num_qubits - 1):
            circuit.cx(i, i + 1)
        
        # Measure correlations
        circuit.measure_all()
        
        # Execute circuit
        job = execute(circuit, self.backend, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        # Analyze correlation patterns
        correlation_analysis = self._analyze_quantum_correlations(counts, portfolio)
        
        return correlation_analysis
    
    def _analyze_quantum_correlations(self, counts, portfolio):
        """
        Analyze quantum correlation patterns
        """
        correlation_patterns = {}
        
        for state, count in counts.items():
            if count > 10:  # Significant patterns
                # Analyze which assets default together
                defaulting_assets = []
                for i, bit in enumerate(state):
                    if i < len(portfolio) and bit == '1':
                        defaulting_assets.append(portfolio.iloc[i]['asset_id'])
                
                if len(defaulting_assets) > 1:
                    pattern_key = tuple(sorted(defaulting_assets))
                    correlation_patterns[pattern_key] = count
        
        return correlation_patterns

def compare_portfolio_models():
    """
    Compare classical and quantum portfolio credit risk models
    """
    # Create portfolio
    classical_model = ClassicalPortfolioCreditRisk()
    portfolio = classical_model.create_portfolio(n_assets=8)
    correlation_matrix = classical_model.create_correlation_matrix(len(portfolio))
    
    print("Portfolio Overview:")
    print(portfolio[['asset_id', 'exposure', 'pd', 'lgd', 'sector']])
    
    # Classical analysis
    print("\n=== Classical Portfolio Analysis ===")
    expected_loss = classical_model.calculate_expected_loss()
    unexpected_loss = classical_model.calculate_unexpected_loss()
    
    print(f"Expected Loss: ${expected_loss:,.2f}")
    print(f"Unexpected Loss: ${unexpected_loss:,.2f}")
    
    # Classical Monte Carlo
    classical_losses = classical_model.monte_carlo_simulation(n_simulations=5000)
    classical_risk_measures = classical_model.calculate_risk_measures(classical_losses)
    
    print("\nClassical Risk Measures:")
    for measure, value in classical_risk_measures.items():
        print(f"{measure}: ${value:,.2f}")
    
    # Quantum analysis
    print("\n=== Quantum Portfolio Analysis ===")
    quantum_model = QuantumPortfolioCreditRisk(num_qubits=8)
    quantum_losses = quantum_model.quantum_monte_carlo_simulation(portfolio, n_simulations=500)
    quantum_risk_measures = classical_model.calculate_risk_measures(quantum_losses)
    
    print("Quantum Risk Measures:")
    for measure, value in quantum_risk_measures.items():
        print(f"{measure}: ${value:,.2f}")
    
    # Quantum correlation analysis
    quantum_correlations = quantum_model.quantum_correlation_analysis(portfolio)
    
    print("\nQuantum Correlation Patterns:")
    for pattern, count in sorted(quantum_correlations.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"Assets {pattern}: {count} occurrences")
    
    # Compare results
    print("\n=== Comparison ===")
    comparison_data = []
    
    for measure in ['expected_loss', 'unexpected_loss', 'var_95', 'var_99']:
        classical_value = classical_risk_measures[measure]
        quantum_value = quantum_risk_measures[measure]
        
        diff_pct = abs(classical_value - quantum_value) / classical_value * 100
        
        print(f"{measure}:")
        print(f"  Classical: ${classical_value:,.2f}")
        print(f"  Quantum: ${quantum_value:,.2f}")
        print(f"  Difference: {diff_pct:.2f}%")
        
        comparison_data.append({
            'measure': measure,
            'classical': classical_value,
            'quantum': quantum_value,
            'difference_pct': diff_pct
        })
    
    # Plot comparison
    plot_portfolio_comparison(comparison_data, classical_losses, quantum_losses)
    
    return classical_risk_measures, quantum_risk_measures, quantum_correlations

def plot_portfolio_comparison(comparison_data, classical_losses, quantum_losses):
    """
    Plot portfolio comparison results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Risk measures comparison
    measures = [d['measure'] for d in comparison_data]
    classical_values = [d['classical'] for d in comparison_data]
    quantum_values = [d['quantum'] for d in comparison_data]
    
    x = np.arange(len(measures))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, classical_values, width, label='Classical', alpha=0.7)
    axes[0, 0].bar(x + width/2, quantum_values, width, label='Quantum', alpha=0.7)
    axes[0, 0].set_xlabel('Risk Measures')
    axes[0, 0].set_ylabel('Value ($)')
    axes[0, 0].set_title('Risk Measures: Classical vs Quantum')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(measures, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Loss distributions
    axes[0, 1].hist(classical_losses, bins=50, alpha=0.7, label='Classical', density=True)
    axes[0, 1].hist(quantum_losses, bins=50, alpha=0.7, label='Quantum', density=True)
    axes[0, 1].set_xlabel('Portfolio Loss')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Portfolio Loss Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Difference analysis
    differences = [d['difference_pct'] for d in comparison_data]
    axes[1, 0].bar(measures, differences, alpha=0.7, color='orange')
    axes[1, 0].set_xlabel('Risk Measures')
    axes[1, 0].set_ylabel('Difference (%)')
    axes[1, 0].set_title('Quantum vs Classical Difference')
    axes[1, 0].set_xticklabels(measures, rotation=45)
    axes[1, 0].grid(True)
    
    # Plot 4: Correlation heatmap (simplified)
    correlation_data = np.random.rand(8, 8)  # Simplified for demo
    correlation_data = (correlation_data + correlation_data.T) / 2
    np.fill_diagonal(correlation_data, 1.0)
    
    im = axes[1, 1].imshow(correlation_data, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 1].set_title('Quantum Correlation Matrix')
    axes[1, 1].set_xlabel('Asset Index')
    axes[1, 1].set_ylabel('Asset Index')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()

# Exercise: Quantum Portfolio Optimization
def quantum_portfolio_optimization_exercise():
    """
    Exercise: Implement quantum portfolio optimization
    """
    # Create portfolio
    classical_model = ClassicalPortfolioCreditRisk()
    portfolio = classical_model.create_portfolio(n_assets=6)
    
    print("=== Quantum Portfolio Optimization ===")
    
    # Define optimization problem
    # Objective: Minimize risk while maintaining expected return
    target_return = 0.05  # 5% target return
    
    # Quantum optimization using QAOA
    num_qubits = 6
    
    # Create cost function
    def cost_function(weights):
        # Calculate portfolio risk
        portfolio_risk = calculate_portfolio_risk(weights, portfolio)
        
        # Calculate portfolio return
        portfolio_return = calculate_portfolio_return(weights, portfolio)
        
        # Penalty for not meeting target return
        return_penalty = max(0, target_return - portfolio_return) * 1000
        
        return portfolio_risk + return_penalty
    
    # Quantum optimization
    optimizer = SPSA(maxiter=100)
    
    # Initial weights
    initial_weights = np.ones(6) / 6
    
    # Optimize
    result = minimize(
        cost_function,
        initial_weights,
        method='SLSQP',
        constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        bounds=[(0, 1)] * 6
    )
    
    optimal_weights = result.x
    
    print("Optimal Portfolio Weights:")
    for i, weight in enumerate(optimal_weights):
        print(f"Asset {i}: {weight:.4f}")
    
    print(f"\nPortfolio Risk: {cost_function(optimal_weights):.4f}")
    print(f"Portfolio Return: {calculate_portfolio_return(optimal_weights, portfolio):.4f}")
    
    return optimal_weights

def calculate_portfolio_risk(weights, portfolio):
    """
    Calculate portfolio risk
    """
    # Simplified risk calculation
    individual_risks = portfolio['pd'] * portfolio['lgd'] * portfolio['exposure']
    portfolio_risk = np.sum(weights * individual_risks)
    
    return portfolio_risk

def calculate_portfolio_return(weights, portfolio):
    """
    Calculate portfolio return
    """
    # Simplified return calculation
    # Assume return is inversely related to risk
    individual_returns = 1 - (portfolio['pd'] * portfolio['lgd'])
    portfolio_return = np.sum(weights * individual_returns)
    
    return portfolio_return

# Exercise: Quantum Risk Decomposition
def quantum_risk_decomposition_exercise():
    """
    Exercise: Implement quantum risk decomposition
    """
    # Create portfolio
    classical_model = ClassicalPortfolioCreditRisk()
    portfolio = classical_model.create_portfolio(n_assets=6)
    
    print("=== Quantum Risk Decomposition ===")
    
    # Quantum circuit for risk decomposition
    num_qubits = 6
    circuit = QuantumCircuit(num_qubits, num_qubits)
    
    # Encode risk factors
    for i, (_, asset) in enumerate(portfolio.iterrows()):
        # Encode default risk
        default_risk = asset['pd']
        circuit.rx(default_risk * np.pi, i)
        
        # Encode loss severity
        loss_severity = asset['lgd']
        circuit.ry(loss_severity * np.pi, i)
        
        # Encode exposure
        exposure_risk = min(asset['exposure'] / 1000000, 1.0)
        circuit.rz(exposure_risk * np.pi, i)
    
    # Add risk factor interactions
    circuit.cx(0, 1)  # Default risk correlation
    circuit.cx(1, 2)  # Loss severity correlation
    circuit.cx(2, 3)  # Exposure correlation
    circuit.cx(3, 4)  # Cross-factor correlation
    circuit.cx(4, 5)  # Systemic correlation
    
    # Measure risk components
    circuit.measure_all()
    
    # Execute circuit
    backend = Aer.get_backend('qasm_simulator')
    job = execute(circuit, backend, shots=1000)
    result = job.result()
    counts = result.get_counts()
    
    # Analyze risk decomposition
    risk_components = analyze_risk_components(counts, portfolio)
    
    print("Risk Decomposition Results:")
    for component, value in risk_components.items():
        print(f"{component}: {value:.4f}")
    
    return risk_components

def analyze_risk_components(counts, portfolio):
    """
    Analyze risk components from quantum measurements
    """
    risk_components = {
        'default_risk': 0,
        'loss_severity_risk': 0,
        'exposure_risk': 0,
        'correlation_risk': 0,
        'systemic_risk': 0
    }
    
    total_shots = sum(counts.values())
    
    for state, count in counts.items():
        # Analyze each qubit for different risk types
        for i, bit in enumerate(state):
            if i < len(portfolio):
                asset = portfolio.iloc[i]
                
                if bit == '1':
                    # Default risk contribution
                    risk_components['default_risk'] += asset['pd'] * count / total_shots
                    
                    # Loss severity contribution
                    risk_components['loss_severity_risk'] += asset['lgd'] * count / total_shots
                    
                    # Exposure contribution
                    exposure_norm = asset['exposure'] / 1000000
                    risk_components['exposure_risk'] += exposure_norm * count / total_shots
    
    # Calculate correlation and systemic risk
    risk_components['correlation_risk'] = np.mean(list(risk_components.values())[:3]) * 0.3
    risk_components['systemic_risk'] = np.mean(list(risk_components.values())[:3]) * 0.2
    
    return risk_components

# Run the main comparison
if __name__ == "__main__":
    print("Comparing Classical vs Quantum Portfolio Credit Risk Models...")
    classical_measures, quantum_measures, quantum_correlations = compare_portfolio_models()
    
    print("\nQuantum Portfolio Optimization Exercise:")
    optimal_weights = quantum_portfolio_optimization_exercise()
    
    print("\nQuantum Risk Decomposition Exercise:")
    risk_components = quantum_risk_decomposition_exercise()
```

## 📊 Kết quả và Phân tích

### **Performance Comparison:**

#### **Classical Portfolio Models:**
- **CreditMetrics**: Industry standard, well-understood
- **Credit Risk+**: Analytical solution, fast computation
- **Monte Carlo**: Flexible, computationally intensive
- **Correlation Matrix**: N² parameters, static assumptions

#### **Quantum Portfolio Models:**
- **Quantum Monte Carlo**: Potential speedup, true randomness
- **Quantum Optimization**: Better solutions cho complex problems
- **Quantum Correlation**: Natural modeling of dependencies
- **Quantum Risk Decomposition**: Granular risk analysis

### **Key Insights:**

#### **1. Quantum Advantages:**
- **Correlation Modeling**: Entanglement naturally captures dependencies
- **Optimization**: Quantum algorithms find better solutions
- **Risk Decomposition**: Quantum measurements provide detailed risk breakdown
- **Uncertainty**: Quantum measurements provide uncertainty estimates

#### **2. Classical Advantages:**
- **Maturity**: Well-established methods và tools
- **Interpretability**: Clear risk decomposition
- **Speed**: Fast computation cho simple models
- **Regulatory Acceptance**: Accepted by regulators

#### **3. Hybrid Approach:**
- Use quantum cho complex correlations
- Use classical cho simple calculations
- Combine both approaches cho optimal results

## 🎯 Bài tập về nhà

### **Exercise 1: Quantum Capital Allocation**
Implement quantum capital allocation system cho portfolio management.

### **Exercise 2: Quantum Portfolio Rebalancing**
Build quantum portfolio rebalancing algorithm.

### **Exercise 3: Quantum Regulatory Compliance**
Implement quantum models cho regulatory requirements.

### **Exercise 4: Quantum Portfolio Stress Testing**
Create quantum stress testing framework cho portfolios.

---

> *"Quantum computing enables more sophisticated portfolio risk modeling by naturally capturing complex correlations and dependencies."* - Quantum Finance Research

> Ngày tiếp theo: [Stress Testing và Scenario Analysis](Day5.md) 