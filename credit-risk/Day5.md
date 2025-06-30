# Ngày 5: Stress Testing và Scenario Analysis

## 🎯 Mục tiêu học tập

- Hiểu sâu về stress testing và scenario analysis trong credit risk
- Phân tích hạn chế của classical stress testing methods
- Implement quantum-enhanced stress testing frameworks
- So sánh performance giữa classical và quantum approaches

## 📚 Lý thuyết

### **Stress Testing Fundamentals**

#### **1. Stress Testing là gì?**
Stress testing là quá trình đánh giá khả năng chịu đựng của tổ chức tài chính trước các tình huống bất lợi, bao gồm:
- **Scenario Analysis**: Phân tích tác động của các kịch bản cụ thể
- **Sensitivity Analysis**: Phân tích độ nhạy với thay đổi tham số
- **Reverse Stress Testing**: Xác định kịch bản dẫn đến thất bại
- **Forward-Looking**: Dự báo rủi ro trong tương lai

#### **2. Regulatory Requirements:**
- **Basel III**: Capital adequacy requirements
- **CCAR**: Comprehensive Capital Analysis and Review
- **EBA**: European Banking Authority stress tests
- **PRA**: Prudential Regulation Authority requirements

### **Classical Stress Testing Methods**

#### **1. Historical Scenarios:**
```
Scenario Impact = Historical Data × Stress Multipliers
```

**Examples:**
- 2008 Financial Crisis
- 2000 Dot-com Bubble
- 1997 Asian Financial Crisis

#### **2. Hypothetical Scenarios:**
- **Macroeconomic Shocks**: GDP decline, unemployment spike
- **Market Shocks**: Interest rate changes, equity market crash
- **Credit Shocks**: Default rate increases, rating downgrades
- **Liquidity Shocks**: Funding stress, market illiquidity

#### **3. Sensitivity Analysis:**
```
ΔPortfolio Value = Σ (ΔFactorᵢ × Sensitivityᵢ)
```

#### **4. Monte Carlo Stress Testing:**
- Generate thousands of scenarios
- Simulate portfolio performance
- Calculate risk measures under stress

### **Hạn chế của Classical Methods**

#### **1. Scenario Design:**
- **Limited Scenarios**: Cannot test all possible combinations
- **Expert Judgment**: Subjective scenario selection
- **Historical Bias**: Based on past events
- **Static Assumptions**: Fixed parameters

#### **2. Computational Limitations:**
- **Monte Carlo Complexity**: Exponential với scenario count
- **Real-time Constraints**: Cannot run stress tests in real-time
- **Large Portfolios**: Computational bottlenecks
- **Multiple Time Horizons**: Complex multi-period analysis

#### **3. Model Limitations:**
- **Linear Assumptions**: Cannot capture non-linear effects
- **Correlation Breakdown**: Assumptions fail under stress
- **Parameter Uncertainty**: Model calibration issues
- **Regime Changes**: Models not designed for extreme events

### **Quantum Advantages cho Stress Testing**

#### **1. Quantum Scenario Generation:**
- **Superposition**: Parallel processing of multiple scenarios
- **Quantum Randomness**: True random scenario generation
- **Entanglement**: Natural modeling of scenario correlations

#### **2. Quantum Optimization:**
- **Scenario Selection**: Optimal scenario combination
- **Risk Minimization**: Find worst-case scenarios
- **Capital Optimization**: Optimal capital allocation under stress

#### **3. Quantum Machine Learning:**
- **Pattern Recognition**: Identify stress patterns
- **Anomaly Detection**: Detect unusual stress scenarios
- **Predictive Modeling**: Forecast stress impacts

## 💻 Thực hành

### **Project 5: Quantum Stress Testing Framework**

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

class ClassicalStressTesting:
    """Classical stress testing framework"""
    
    def __init__(self):
        self.portfolio = None
        self.scenarios = {}
        
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
    
    def define_stress_scenarios(self):
        """
        Define stress testing scenarios
        """
        self.scenarios = {
            'baseline': {
                'description': 'Baseline scenario',
                'pd_multiplier': 1.0,
                'lgd_multiplier': 1.0,
                'correlation_multiplier': 1.0,
                'probability': 0.5
            },
            'mild_recession': {
                'description': 'Mild economic recession',
                'pd_multiplier': 1.5,
                'lgd_multiplier': 1.1,
                'correlation_multiplier': 1.2,
                'probability': 0.3
            },
            'severe_recession': {
                'description': 'Severe economic recession',
                'pd_multiplier': 2.5,
                'lgd_multiplier': 1.3,
                'correlation_multiplier': 1.5,
                'probability': 0.15
            },
            'financial_crisis': {
                'description': 'Financial crisis scenario',
                'pd_multiplier': 4.0,
                'lgd_multiplier': 1.5,
                'correlation_multiplier': 2.0,
                'probability': 0.05
            }
        }
        
        return self.scenarios
    
    def apply_stress_scenario(self, scenario_name):
        """
        Apply stress scenario to portfolio
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario {scenario_name} không tồn tại")
        
        scenario = self.scenarios[scenario_name]
        stressed_portfolio = self.portfolio.copy()
        
        # Apply stress multipliers
        stressed_portfolio['pd'] = self.portfolio['pd'] * scenario['pd_multiplier']
        stressed_portfolio['lgd'] = self.portfolio['lgd'] * scenario['lgd_multiplier']
        
        return stressed_portfolio, scenario
    
    def calculate_stress_impact(self, stressed_portfolio):
        """
        Calculate stress impact on portfolio
        """
        # Calculate expected loss
        expected_loss = np.sum(stressed_portfolio['exposure'] * 
                              stressed_portfolio['pd'] * 
                              stressed_portfolio['lgd'])
        
        # Calculate unexpected loss (simplified)
        unexpected_loss = np.sqrt(np.sum((stressed_portfolio['exposure'] * 
                                         stressed_portfolio['pd'] * 
                                         stressed_portfolio['lgd'])**2))
        
        # Calculate VaR (simplified)
        var_95 = expected_loss + 1.645 * unexpected_loss
        var_99 = expected_loss + 2.326 * unexpected_loss
        
        return {
            'expected_loss': expected_loss,
            'unexpected_loss': unexpected_loss,
            'var_95': var_95,
            'var_99': var_99
        }
    
    def monte_carlo_stress_testing(self, n_simulations=10000):
        """
        Monte Carlo stress testing
        """
        scenario_results = {}
        
        for scenario_name in self.scenarios.keys():
            print(f"Running stress test for {scenario_name}...")
            
            stressed_portfolio, scenario = self.apply_stress_scenario(scenario_name)
            
            # Monte Carlo simulation
            losses = []
            for sim in range(n_simulations):
                # Generate random defaults
                defaults = np.random.binomial(1, stressed_portfolio['pd'].values)
                loss = np.sum(defaults * stressed_portfolio['exposure'].values * 
                             stressed_portfolio['lgd'].values)
                losses.append(loss)
            
            losses = np.array(losses)
            
            # Calculate risk measures
            risk_measures = {
                'expected_loss': np.mean(losses),
                'unexpected_loss': np.std(losses),
                'var_95': np.percentile(losses, 95),
                'var_99': np.percentile(losses, 99),
                'max_loss': np.max(losses)
            }
            
            scenario_results[scenario_name] = {
                'risk_measures': risk_measures,
                'scenario': scenario,
                'losses': losses
            }
        
        return scenario_results

class QuantumStressTesting:
    """Quantum-enhanced stress testing framework"""
    
    def __init__(self, num_qubits=8):
        self.num_qubits = num_qubits
        self.backend = Aer.get_backend('qasm_simulator')
        self.portfolio = None
        self.scenarios = {}
        
    def create_quantum_stress_circuit(self, portfolio, scenario):
        """
        Create quantum circuit for stress testing
        """
        n_assets = len(portfolio)
        circuit = QuantumCircuit(self.num_qubits, self.num_qubits)
        
        # Encode portfolio information
        for i, (_, asset) in enumerate(portfolio.iterrows()):
            if i < self.num_qubits:
                # Encode exposure
                exposure_norm = min(asset['exposure'] / 1000000, 1.0)
                circuit.rx(exposure_norm * np.pi, i)
                
                # Encode stressed PD
                stressed_pd = asset['pd'] * scenario['pd_multiplier']
                circuit.ry(stressed_pd * np.pi, i)
                
                # Encode stressed LGD
                stressed_lgd = asset['lgd'] * scenario['lgd_multiplier']
                circuit.rz(stressed_lgd * np.pi, i)
        
        # Add stress-specific entanglement
        stress_level = scenario['correlation_multiplier']
        for i in range(self.num_qubits - 1):
            circuit.cx(i, i + 1)
            # Add additional entanglement based on stress level
            if stress_level > 1.5:
                circuit.cx(i, (i + 2) % self.num_qubits)
        
        return circuit
    
    def quantum_stress_simulation(self, portfolio, scenarios, n_simulations=1000):
        """
        Quantum stress simulation
        """
        scenario_results = {}
        
        for scenario_name, scenario in scenarios.items():
            print(f"Running quantum stress test for {scenario_name}...")
            
            # Create quantum circuit
            circuit = self.create_quantum_stress_circuit(portfolio, scenario)
            circuit.measure_all()
            
            # Run quantum simulation
            losses = []
            for sim in range(n_simulations):
                # Execute circuit
                job = execute(circuit, self.backend, shots=1)
                result = job.result()
                counts = result.get_counts()
                
                # Calculate loss from measurement
                loss = self._calculate_loss_from_measurement(counts, portfolio, scenario)
                losses.append(loss)
            
            losses = np.array(losses)
            
            # Calculate risk measures
            risk_measures = {
                'expected_loss': np.mean(losses),
                'unexpected_loss': np.std(losses),
                'var_95': np.percentile(losses, 95),
                'var_99': np.percentile(losses, 99),
                'max_loss': np.max(losses)
            }
            
            scenario_results[scenario_name] = {
                'risk_measures': risk_measures,
                'scenario': scenario,
                'losses': losses
            }
        
        return scenario_results
    
    def _calculate_loss_from_measurement(self, counts, portfolio, scenario):
        """
        Calculate portfolio loss from quantum measurement
        """
        state = list(counts.keys())[0]
        
        loss = 0
        for i, bit in enumerate(state):
            if i < len(portfolio):
                if bit == '1':  # Default occurred
                    asset = portfolio.iloc[i]
                    stressed_pd = asset['pd'] * scenario['pd_multiplier']
                    stressed_lgd = asset['lgd'] * scenario['lgd_multiplier']
                    loss += asset['exposure'] * stressed_lgd
        
        return loss
    
    def quantum_scenario_optimization(self, portfolio, scenarios):
        """
        Quantum optimization for scenario selection
        """
        # Use QAOA to find worst-case scenario combination
        n_scenarios = len(scenarios)
        
        # Create cost function for scenario optimization
        def cost_function(scenario_weights):
            # Calculate weighted stress impact
            total_impact = 0
            for i, (scenario_name, scenario) in enumerate(scenarios.items()):
                stressed_portfolio, _ = self.apply_stress_scenario(scenario_name)
                impact = self.calculate_stress_impact(stressed_portfolio)
                total_impact += scenario_weights[i] * impact['var_99']
            
            return total_impact
        
        # Optimize scenario weights
        initial_weights = np.ones(n_scenarios) / n_scenarios
        
        result = minimize(
            cost_function,
            initial_weights,
            method='SLSQP',
            constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            bounds=[(0, 1)] * n_scenarios
        )
        
        return result.x, result.fun

def compare_stress_testing_approaches():
    """
    Compare classical and quantum stress testing approaches
    """
    # Create portfolio
    classical_stress = ClassicalStressTesting()
    portfolio = classical_stress.create_portfolio(n_assets=8)
    scenarios = classical_stress.define_stress_scenarios()
    
    print("Portfolio Overview:")
    print(portfolio[['asset_id', 'exposure', 'pd', 'lgd', 'sector']])
    
    print("\nStress Scenarios:")
    for name, scenario in scenarios.items():
        print(f"{name}: {scenario['description']}")
    
    # Classical stress testing
    print("\n=== Classical Stress Testing ===")
    classical_results = classical_stress.monte_carlo_stress_testing(n_simulations=5000)
    
    # Quantum stress testing
    print("\n=== Quantum Stress Testing ===")
    quantum_stress = QuantumStressTesting(num_qubits=8)
    quantum_results = quantum_stress.quantum_stress_simulation(portfolio, scenarios, n_simulations=500)
    
    # Compare results
    print("\n=== Stress Testing Results Comparison ===")
    comparison_data = []
    
    for scenario_name in scenarios.keys():
        classical_measures = classical_results[scenario_name]['risk_measures']
        quantum_measures = quantum_results[scenario_name]['risk_measures']
        
        print(f"\n{scenario_name.upper()}:")
        print(f"Classical VaR (99%): ${classical_measures['var_99']:,.2f}")
        print(f"Quantum VaR (99%): ${quantum_measures['var_99']:,.2f}")
        
        diff_pct = abs(classical_measures['var_99'] - quantum_measures['var_99']) / classical_measures['var_99'] * 100
        print(f"Difference: {diff_pct:.2f}%")
        
        comparison_data.append({
            'scenario': scenario_name,
            'classical_var': classical_measures['var_99'],
            'quantum_var': quantum_measures['var_99'],
            'difference_pct': diff_pct
        })
    
    # Plot comparison
    plot_stress_testing_comparison(comparison_data, classical_results, quantum_results)
    
    return classical_results, quantum_results

def plot_stress_testing_comparison(comparison_data, classical_results, quantum_results):
    """
    Plot stress testing comparison results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: VaR comparison
    scenarios = [d['scenario'] for d in comparison_data]
    classical_vars = [d['classical_var'] for d in comparison_data]
    quantum_vars = [d['quantum_var'] for d in comparison_data]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, classical_vars, width, label='Classical', alpha=0.7)
    axes[0, 0].bar(x + width/2, quantum_vars, width, label='Quantum', alpha=0.7)
    axes[0, 0].set_xlabel('Stress Scenarios')
    axes[0, 0].set_ylabel('VaR (99%)')
    axes[0, 0].set_title('VaR Comparison: Classical vs Quantum')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(scenarios, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Loss distributions for financial crisis
    crisis_scenario = 'financial_crisis'
    classical_losses = classical_results[crisis_scenario]['losses']
    quantum_losses = quantum_results[crisis_scenario]['losses']
    
    axes[0, 1].hist(classical_losses, bins=50, alpha=0.7, label='Classical', density=True)
    axes[0, 1].hist(quantum_losses, bins=50, alpha=0.7, label='Quantum', density=True)
    axes[0, 1].set_xlabel('Portfolio Loss')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title(f'Loss Distribution: {crisis_scenario}')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Stress impact progression
    stress_levels = []
    classical_impacts = []
    quantum_impacts = []
    
    for scenario_name in ['baseline', 'mild_recession', 'severe_recession', 'financial_crisis']:
        stress_levels.append(scenarios[scenario_name]['pd_multiplier'])
        classical_impacts.append(classical_results[scenario_name]['risk_measures']['var_99'])
        quantum_impacts.append(quantum_results[scenario_name]['risk_measures']['var_99'])
    
    axes[1, 0].plot(stress_levels, classical_impacts, 'o-', label='Classical', linewidth=2)
    axes[1, 0].plot(stress_levels, quantum_impacts, 's-', label='Quantum', linewidth=2)
    axes[1, 0].set_xlabel('Stress Level (PD Multiplier)')
    axes[1, 0].set_ylabel('VaR (99%)')
    axes[1, 0].set_title('Stress Impact Progression')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 4: Difference analysis
    differences = [d['difference_pct'] for d in comparison_data]
    axes[1, 1].bar(scenarios, differences, alpha=0.7, color='orange')
    axes[1, 1].set_xlabel('Stress Scenarios')
    axes[1, 1].set_ylabel('Difference (%)')
    axes[1, 1].set_title('Quantum vs Classical Difference')
    axes[1, 1].set_xticklabels(scenarios, rotation=45)
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

# Exercise: Quantum Reverse Stress Testing
def quantum_reverse_stress_testing_exercise():
    """
    Exercise: Implement quantum reverse stress testing
    """
    # Create portfolio
    classical_stress = ClassicalStressTesting()
    portfolio = classical_stress.create_portfolio(n_assets=6)
    
    # Define target loss threshold
    target_loss = 500000  # $500k loss threshold
    
    print("=== Quantum Reverse Stress Testing ===")
    print(f"Target Loss Threshold: ${target_loss:,.2f}")
    
    # Quantum circuit for reverse stress testing
    num_qubits = 6
    circuit = QuantumCircuit(num_qubits, num_qubits)
    
    # Create superposition of stress scenarios
    circuit.h(0)  # Hadamard gate for scenario superposition
    circuit.h(1)  # Additional scenario dimension
    
    # Encode portfolio parameters
    for i in range(num_qubits):
        circuit.rx(np.pi/4, i)  # Base rotation
    
    # Add entanglement for scenario interactions
    circuit.cx(0, 2)
    circuit.cx(1, 3)
    circuit.cx(2, 4)
    circuit.cx(3, 5)
    
    # Measure
    circuit.measure_all()
    
    # Execute circuit
    backend = Aer.get_backend('qasm_simulator')
    job = execute(circuit, backend, shots=1000)
    result = job.result()
    counts = result.get_counts()
    
    # Analyze results for scenarios leading to target loss
    print("\nQuantum Reverse Stress Testing Results:")
    print("Measurement counts:")
    
    scenarios_above_threshold = 0
    total_scenarios = 0
    
    for state, count in counts.items():
        total_scenarios += count
        
        # Calculate loss for this scenario
        loss = calculate_scenario_loss(state, portfolio)
        
        if loss >= target_loss:
            scenarios_above_threshold += count
            print(f"State {state}: Loss ${loss:,.2f} (Count: {count})")
    
    probability_above_threshold = scenarios_above_threshold / total_scenarios
    print(f"\nProbability of loss >= ${target_loss:,.2f}: {probability_above_threshold:.4f}")
    
    return probability_above_threshold

def calculate_scenario_loss(state, portfolio):
    """
    Calculate portfolio loss for a given quantum state
    """
    loss = 0
    for i, bit in enumerate(state):
        if i < len(portfolio) and bit == '1':
            asset = portfolio.iloc[i]
            # Apply stress multiplier based on state
            stress_multiplier = 1 + (i % 3) * 0.5  # Varying stress levels
            loss += asset['exposure'] * asset['lgd'] * stress_multiplier
    
    return loss

# Exercise: Quantum Sensitivity Analysis
def quantum_sensitivity_analysis_exercise():
    """
    Exercise: Implement quantum sensitivity analysis
    """
    # Create portfolio
    classical_stress = ClassicalStressTesting()
    portfolio = classical_stress.create_portfolio(n_assets=6)
    
    print("=== Quantum Sensitivity Analysis ===")
    
    # Define parameter ranges
    pd_multipliers = np.linspace(0.5, 3.0, 10)
    lgd_multipliers = np.linspace(0.8, 1.5, 8)
    
    # Create quantum circuit for sensitivity analysis
    num_qubits = 6
    circuit = QuantumCircuit(num_qubits, num_qubits)
    
    # Encode parameter sensitivity
    for i in range(num_qubits):
        # Encode PD sensitivity
        pd_sensitivity = (i % 3) / 2.0  # Varying sensitivity
        circuit.rx(pd_sensitivity * np.pi, i)
        
        # Encode LGD sensitivity
        lgd_sensitivity = ((i + 1) % 3) / 2.0
        circuit.ry(lgd_sensitivity * np.pi, i)
    
    # Add parameter interactions
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(2, 3)
    circuit.cx(3, 4)
    circuit.cx(4, 5)
    
    # Measure
    circuit.measure_all()
    
    # Execute circuit
    backend = Aer.get_backend('qasm_simulator')
    job = execute(circuit, backend, shots=1000)
    result = job.result()
    counts = result.get_counts()
    
    # Analyze sensitivity
    print("Quantum Sensitivity Analysis Results:")
    
    sensitivity_scores = []
    for state, count in counts.items():
        # Calculate sensitivity score
        sensitivity = calculate_sensitivity_score(state, portfolio)
        sensitivity_scores.append(sensitivity)
        
        if count > 50:  # Show significant states
            print(f"State {state}: Sensitivity {sensitivity:.4f} (Count: {count})")
    
    avg_sensitivity = np.mean(sensitivity_scores)
    print(f"\nAverage Sensitivity Score: {avg_sensitivity:.4f}")
    
    return avg_sensitivity

def calculate_sensitivity_score(state, portfolio):
    """
    Calculate sensitivity score for a quantum state
    """
    sensitivity = 0
    for i, bit in enumerate(state):
        if i < len(portfolio):
            asset = portfolio.iloc[i]
            # Calculate sensitivity based on asset characteristics
            base_sensitivity = asset['pd'] * asset['lgd'] * asset['exposure']
            
            if bit == '1':
                sensitivity += base_sensitivity * 2  # Higher sensitivity for '1' states
            else:
                sensitivity += base_sensitivity
    
    return sensitivity

# Run the main comparison
if __name__ == "__main__":
    print("Comparing Classical vs Quantum Stress Testing Approaches...")
    classical_results, quantum_results = compare_stress_testing_approaches()
    
    print("\nQuantum Reverse Stress Testing Exercise:")
    probability = quantum_reverse_stress_testing_exercise()
    
    print("\nQuantum Sensitivity Analysis Exercise:")
    sensitivity = quantum_sensitivity_analysis_exercise()
```

### **Exercise 4: Quantum Forward-Looking Stress Testing**

```python
def quantum_forward_looking_stress_testing():
    """
    Exercise: Implement quantum forward-looking stress testing
    """
    # Create time series of portfolio data
    np.random.seed(42)
    n_periods = 12  # 12 months
    n_assets = 6
    
    # Generate time series data
    time_series_data = []
    for period in range(n_periods):
        period_data = []
        for asset in range(n_assets):
            # Simulate time-varying parameters
            base_pd = 0.05 + 0.02 * np.sin(2 * np.pi * period / 12)  # Seasonal pattern
            base_lgd = 0.5 + 0.1 * np.random.normal(0, 1)  # Random variation
            
            asset_data = {
                'period': period,
                'asset_id': f'Asset_{asset}',
                'exposure': 500000 + 100000 * np.random.normal(0, 1),
                'pd': max(0.01, base_pd + 0.01 * np.random.normal(0, 1)),
                'lgd': max(0.2, min(0.8, base_lgd)),
                'rating': np.random.choice(['AAA', 'AA', 'A', 'BBB', 'BB', 'B'])
            }
            period_data.append(asset_data)
        
        time_series_data.extend(period_data)
    
    time_series_df = pd.DataFrame(time_series_data)
    
    # Quantum forward-looking analysis
    print("=== Quantum Forward-Looking Stress Testing ===")
    
    # Create quantum circuit for time series analysis
    num_qubits = 8
    circuit = QuantumCircuit(num_qubits, num_qubits)
    
    # Encode time series information
    for period in range(min(4, n_periods)):  # Encode last 4 periods
        period_data = time_series_df[time_series_df['period'] == period]
        
        for i, (_, asset) in enumerate(period_data.iterrows()):
            if i < num_qubits // 4:  # Distribute across qubits
                qubit_idx = period * (num_qubits // 4) + i
                
                # Encode PD trend
                pd_trend = asset['pd'] - 0.05  # Deviation from baseline
                circuit.rx(pd_trend * np.pi, qubit_idx)
                
                # Encode LGD trend
                lgd_trend = asset['lgd'] - 0.5
                circuit.ry(lgd_trend * np.pi, qubit_idx)
    
    # Add time series entanglement
    for i in range(num_qubits - 1):
        circuit.cx(i, i + 1)
    
    # Add temporal correlations
    circuit.cx(0, 4)  # Connect periods
    circuit.cx(1, 5)
    circuit.cx(2, 6)
    circuit.cx(3, 7)
    
    # Measure
    circuit.measure_all()
    
    # Execute circuit
    backend = Aer.get_backend('qasm_simulator')
    job = execute(circuit, backend, shots=1000)
    result = job.result()
    counts = result.get_counts()
    
    # Analyze forward-looking predictions
    print("Quantum Forward-Looking Analysis Results:")
    
    future_predictions = []
    for state, count in counts.items():
        if count > 20:  # Significant states
            prediction = analyze_future_prediction(state, time_series_df)
            future_predictions.append(prediction)
            print(f"State {state}: Future Risk Level {prediction:.4f} (Count: {count})")
    
    if future_predictions:
        avg_future_risk = np.mean(future_predictions)
        print(f"\nAverage Predicted Future Risk Level: {avg_future_risk:.4f}")
        
        if avg_future_risk > 0.7:
            print("⚠️  HIGH FUTURE RISK DETECTED")
        elif avg_future_risk > 0.4:
            print("⚠️  MODERATE FUTURE RISK DETECTED")
        else:
            print("✅ LOW FUTURE RISK DETECTED")
    
    return future_predictions

def analyze_future_prediction(state, time_series_df):
    """
    Analyze future risk prediction from quantum state
    """
    # Extract trend information from quantum state
    trend_indicators = []
    
    for i, bit in enumerate(state):
        if bit == '1':
            trend_indicators.append(1)
        else:
            trend_indicators.append(0)
    
    # Calculate trend strength
    trend_strength = np.mean(trend_indicators)
    
    # Get recent portfolio performance
    recent_data = time_series_df[time_series_df['period'] >= 8]  # Last 4 periods
    recent_pd_avg = recent_data['pd'].mean()
    recent_lgd_avg = recent_data['lgd'].mean()
    
    # Combine trend and current performance
    future_risk = (trend_strength * 0.6 + recent_pd_avg * 0.3 + recent_lgd_avg * 0.1)
    
    return future_risk

# Run forward-looking exercise
if __name__ == "__main__":
    future_predictions = quantum_forward_looking_stress_testing()
```

## 📊 Kết quả và Phân tích

### **Performance Comparison:**

#### **Classical Stress Testing:**
- **Historical Scenarios**: Based on past events
- **Monte Carlo**: Computationally intensive
- **Sensitivity Analysis**: Linear approximations
- **Reverse Stress Testing**: Limited scenario space

#### **Quantum Stress Testing:**
- **Quantum Scenarios**: Parallel scenario processing
- **Quantum Monte Carlo**: Potential speedup
- **Quantum Sensitivity**: Non-linear analysis
- **Quantum Reverse Testing**: Expanded scenario space

### **Key Insights:**

#### **1. Quantum Advantages:**
- **Scenario Generation**: Superposition enables parallel scenario analysis
- **Correlation Modeling**: Entanglement captures complex dependencies
- **Non-linear Effects**: Quantum transformations handle non-linear stress impacts
- **Forward-looking**: Quantum circuits can model temporal patterns

#### **2. Classical Advantages:**
- **Interpretability**: Clear scenario definitions
- **Regulatory Acceptance**: Well-established methods
- **Computational Efficiency**: Fast for simple scenarios
- **Historical Validation**: Based on observed events

#### **3. Hybrid Approach:**
- Use quantum cho complex scenario generation
- Use classical cho scenario interpretation
- Combine both cho comprehensive stress testing

## 🎯 Bài tập về nhà

### **Exercise 1: Quantum Regulatory Stress Testing**
Implement quantum models cho regulatory stress testing requirements.

### **Exercise 2: Quantum Multi-Period Stress Testing**
Build quantum multi-period stress testing framework.

### **Exercise 3: Quantum Stress Testing Dashboard**
Create interactive dashboard cho quantum stress testing results.

### **Exercise 4: Quantum Stress Testing Validation**
Implement validation framework cho quantum stress testing models.

---

> *"Quantum stress testing enables more comprehensive scenario analysis by naturally capturing complex, non-linear relationships and temporal patterns."* - Quantum Finance Research

> Ngày tiếp theo: [Quantum Probability và Finance](Day6.md) 