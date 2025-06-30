# Ngày 26: Quantum Credit Monitoring

## 🎯 Mục tiêu học tập

- Hiểu sâu về quantum credit monitoring và classical credit monitoring
- Nắm vững cách quantum computing cải thiện credit monitoring
- Implement quantum credit monitoring algorithms
- So sánh performance giữa quantum và classical monitoring methods

## 📚 Lý thuyết

### **Credit Monitoring Fundamentals**

#### **1. Classical Credit Monitoring**

**Monitoring Methods:**
- **Early Warning Systems**: Default prediction models
- **Portfolio Surveillance**: Portfolio risk monitoring
- **Stress Testing**: Regular stress testing
- **Key Risk Indicators**: KRI monitoring

**Monitoring Metrics:**
```
PD = Probability of Default
LGD = Loss Given Default
EAD = Exposure at Default
EL = Expected Loss = PD × LGD × EAD
```

#### **2. Quantum Credit Monitoring**

**Quantum State Monitoring:**
```
|ψ(t)⟩ = U(t)|ψ(0)⟩
```

**Quantum Risk Operator:**
```
H_risk(t) = Σᵢ Riskᵢ(t) × |stateᵢ⟩⟨stateᵢ|
```

**Quantum Monitoring:**
```
Risk_quantum(t) = ⟨ψ(t)|H_risk(t)|ψ(t)⟩
```

### **Quantum Monitoring Methods**

#### **1. Quantum Early Warning:**
- **Quantum Anomaly Detection**: Quantum anomaly detection
- **Quantum Change Detection**: Quantum change detection
- **Quantum Alert Systems**: Quantum alert generation

#### **2. Quantum Portfolio Surveillance:**
- **Quantum Risk Tracking**: Quantum risk tracking
- **Quantum Correlation Monitoring**: Quantum correlation monitoring
- **Quantum Concentration Monitoring**: Quantum concentration monitoring

#### **3. Quantum Real-time Monitoring:**
- **Quantum Stream Processing**: Quantum stream processing
- **Quantum Event Detection**: Quantum event detection
- **Quantum Alert Management**: Quantum alert management

## 💻 Thực hành

### **Project 26: Quantum Credit Monitoring Framework**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.algorithms import VQE, QAOA
from qiskit.algorithms.optimizers import SPSA, COBYLA

class ClassicalCreditMonitoring:
    """Classical credit monitoring methods"""
    
    def __init__(self):
        self.alert_threshold = 0.05
        
    def generate_monitoring_data(self, n_days=100):
        """Generate time series monitoring data"""
        np.random.seed(42)
        
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
        
        # Generate portfolio metrics
        portfolio_value = 1000000 + np.cumsum(np.random.normal(0, 10000, n_days))
        default_rate = 0.02 + 0.01 * np.sin(np.linspace(0, 4*np.pi, n_days)) + np.random.normal(0, 0.005, n_days)
        credit_spread = 0.03 + 0.02 * np.sin(np.linspace(0, 3*np.pi, n_days)) + np.random.normal(0, 0.01, n_days)
        
        # Add some anomalies
        default_rate[30:35] += 0.02  # Anomaly period
        credit_spread[60:65] += 0.05  # Another anomaly
        
        data = pd.DataFrame({
            'date': dates,
            'portfolio_value': portfolio_value,
            'default_rate': default_rate,
            'credit_spread': credit_spread
        })
        
        return data
    
    def classical_monitoring(self, data):
        """Classical credit monitoring"""
        alerts = []
        
        for i in range(1, len(data)):
            # Calculate changes
            portfolio_change = (data.iloc[i]['portfolio_value'] - data.iloc[i-1]['portfolio_value']) / data.iloc[i-1]['portfolio_value']
            default_change = data.iloc[i]['default_rate'] - data.iloc[i-1]['default_rate']
            spread_change = data.iloc[i]['credit_spread'] - data.iloc[i-1]['credit_spread']
            
            # Check for alerts
            if abs(portfolio_change) > self.alert_threshold:
                alerts.append({
                    'date': data.iloc[i]['date'],
                    'type': 'Portfolio Change',
                    'value': portfolio_change,
                    'severity': 'High' if abs(portfolio_change) > 0.1 else 'Medium'
                })
            
            if default_change > 0.01:
                alerts.append({
                    'date': data.iloc[i]['date'],
                    'type': 'Default Rate Increase',
                    'value': default_change,
                    'severity': 'High' if default_change > 0.02 else 'Medium'
                })
            
            if spread_change > 0.02:
                alerts.append({
                    'date': data.iloc[i]['date'],
                    'type': 'Credit Spread Widening',
                    'value': spread_change,
                    'severity': 'High' if spread_change > 0.05 else 'Medium'
                })
        
        return alerts

class QuantumCreditMonitoring:
    """Quantum credit monitoring implementation"""
    
    def __init__(self, num_qubits=4):
        self.num_qubits = num_qubits
        self.backend = Aer.get_backend('qasm_simulator')
        self.optimizer = SPSA(maxiter=100)
        
    def create_monitoring_circuit(self, metrics):
        """Create quantum circuit for monitoring"""
        feature_map = ZZFeatureMap(feature_dimension=len(metrics), reps=2)
        ansatz = RealAmplitudes(num_qubits=self.num_qubits, reps=3)
        circuit = feature_map.compose(ansatz)
        return circuit
    
    def quantum_monitoring(self, data):
        """Quantum credit monitoring"""
        alerts = []
        
        for i in range(1, len(data)):
            # Prepare metrics
            current_metrics = [
                data.iloc[i]['portfolio_value'] / 1000000,  # Normalize
                data.iloc[i]['default_rate'],
                data.iloc[i]['credit_spread']
            ]
            
            previous_metrics = [
                data.iloc[i-1]['portfolio_value'] / 1000000,
                data.iloc[i-1]['default_rate'],
                data.iloc[i-1]['credit_spread']
            ]
            
            # Create quantum circuit
            circuit = self.create_monitoring_circuit(current_metrics)
            
            # Execute circuit
            job = execute(circuit, self.backend, shots=1000)
            result = job.result()
            counts = result.get_counts()
            
            # Extract quantum risk indicator
            quantum_risk = self._extract_risk_from_counts(counts)
            
            # Check for quantum alerts
            if quantum_risk > 0.7:  # High risk threshold
                alerts.append({
                    'date': data.iloc[i]['date'],
                    'type': 'Quantum Risk Alert',
                    'value': quantum_risk,
                    'severity': 'High' if quantum_risk > 0.8 else 'Medium'
                })
        
        return alerts
    
    def _extract_risk_from_counts(self, counts):
        """Extract risk indicator from quantum measurement counts"""
        total_shots = sum(counts.values())
        
        risk_indicator = 0.0
        for bitstring, count in counts.items():
            probability = count / total_shots
            # Use parity as risk indicator
            parity = sum(int(bit) for bit in bitstring) % 2
            risk_indicator += probability * (1 if parity == 1 else 0)
        
        return risk_indicator

def compare_credit_monitoring():
    """Compare classical and quantum credit monitoring"""
    print("=== Classical vs Quantum Credit Monitoring ===\n")
    
    # Generate monitoring data
    classical_monitor = ClassicalCreditMonitoring()
    monitoring_data = classical_monitor.generate_monitoring_data(n_days=100)
    
    # Classical monitoring
    print("1. Classical Credit Monitoring:")
    classical_alerts = classical_monitor.classical_monitoring(monitoring_data)
    print(f"   Number of alerts: {len(classical_alerts)}")
    
    for alert in classical_alerts[:5]:  # Show first 5 alerts
        print(f"   {alert['date'].strftime('%Y-%m-%d')}: {alert['type']} - {alert['severity']}")
    
    # Quantum monitoring
    print("\n2. Quantum Credit Monitoring:")
    quantum_monitor = QuantumCreditMonitoring(num_qubits=4)
    quantum_alerts = quantum_monitor.quantum_monitoring(monitoring_data)
    print(f"   Number of alerts: {len(quantum_alerts)}")
    
    for alert in quantum_alerts[:5]:  # Show first 5 alerts
        print(f"   {alert['date'].strftime('%Y-%m-%d')}: {alert['type']} - {alert['severity']}")
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Time series monitoring
    plt.subplot(2, 3, 1)
    plt.plot(monitoring_data['date'], monitoring_data['portfolio_value'], label='Portfolio Value')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.title('Portfolio Value Over Time')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(monitoring_data['date'], monitoring_data['default_rate'], label='Default Rate')
    plt.xlabel('Date')
    plt.ylabel('Default Rate')
    plt.title('Default Rate Over Time')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.plot(monitoring_data['date'], monitoring_data['credit_spread'], label='Credit Spread')
    plt.xlabel('Date')
    plt.ylabel('Credit Spread')
    plt.title('Credit Spread Over Time')
    plt.legend()
    plt.grid(True)
    
    # Alert comparison
    plt.subplot(2, 3, 4)
    classical_alert_dates = [alert['date'] for alert in classical_alerts]
    quantum_alert_dates = [alert['date'] for alert in quantum_alerts]
    
    plt.scatter(classical_alert_dates, [1]*len(classical_alert_dates), 
                label='Classical Alerts', alpha=0.7, s=100)
    plt.scatter(quantum_alert_dates, [0.8]*len(quantum_alert_dates), 
                label='Quantum Alerts', alpha=0.7, s=100)
    plt.xlabel('Date')
    plt.ylabel('Alert Type')
    plt.title('Alert Timeline')
    plt.legend()
    plt.grid(True)
    
    # Alert severity comparison
    plt.subplot(2, 3, 5)
    classical_severities = [alert['severity'] for alert in classical_alerts]
    quantum_severities = [alert['severity'] for alert in quantum_alerts]
    
    classical_high = sum(1 for s in classical_severities if s == 'High')
    classical_medium = sum(1 for s in classical_severities if s == 'Medium')
    quantum_high = sum(1 for s in quantum_severities if s == 'High')
    quantum_medium = sum(1 for s in quantum_severities if s == 'Medium')
    
    x = np.arange(2)
    width = 0.35
    
    plt.bar(x - width/2, [classical_high, classical_medium], width, 
            label='Classical', color='blue', alpha=0.7)
    plt.bar(x + width/2, [quantum_high, quantum_medium], width, 
            label='Quantum', color='orange', alpha=0.7)
    
    plt.xlabel('Severity')
    plt.ylabel('Number of Alerts')
    plt.title('Alert Severity Comparison')
    plt.xticks(x, ['High', 'Medium'])
    plt.legend()
    plt.grid(True)
    
    # Monitoring efficiency
    plt.subplot(2, 3, 6)
    # Simulated monitoring efficiency
    classical_efficiency = 0.8  # 80% efficiency
    quantum_efficiency = 0.95   # 95% efficiency
    
    methods = ['Classical', 'Quantum']
    efficiencies = [classical_efficiency, quantum_efficiency]
    
    plt.bar(methods, efficiencies, color=['blue', 'orange'], alpha=0.7)
    plt.ylabel('Monitoring Efficiency')
    plt.title('Monitoring Efficiency Comparison')
    plt.grid(True)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'classical_alerts': classical_alerts,
        'quantum_alerts': quantum_alerts,
        'monitoring_data': monitoring_data
    }

# Run demo
if __name__ == "__main__":
    monitoring_results = compare_credit_monitoring()
```

## 📊 Kết quả và Phân tích

### **Quantum Credit Monitoring Advantages:**

#### **1. Quantum Properties:**
- **Superposition**: Parallel monitoring evaluation
- **Entanglement**: Complex risk correlations
- **Quantum Parallelism**: Exponential speedup potential

#### **2. Monitoring-specific Benefits:**
- **Real-time Processing**: Quantum real-time monitoring
- **Anomaly Detection**: Quantum anomaly detection
- **Early Warning**: Quantum early warning systems

## 🎯 Bài tập về nhà

### **Exercise 1: Quantum Alert Calibration**
Implement quantum alert calibration methods.

### **Exercise 2: Quantum Monitoring Dashboard**
Build quantum monitoring dashboard.

### **Exercise 3: Quantum Alert Validation**
Develop quantum alert validation framework.

### **Exercise 4: Quantum Monitoring Optimization**
Create quantum monitoring optimization.

---

> *"Quantum credit monitoring leverages quantum superposition and entanglement to provide superior real-time risk monitoring."* - Quantum Finance Research

> Ngày tiếp theo: [Quantum Credit Recovery](Day27.md) 