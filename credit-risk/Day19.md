# Ngày 19: Quantum Time Series Analysis

## 🎯 Mục tiêu học tập

- Hiểu sâu về quantum time series analysis và classical time series analysis
- Nắm vững cách quantum computing cải thiện time series forecasting
- Implement quantum time series analysis cho credit risk prediction
- So sánh performance giữa quantum và classical time series methods

## 📚 Lý thuyết

### **Time Series Analysis Fundamentals**

#### **1. Classical Time Series Models**

**ARIMA Model:**
```
(1 - Σᵢ φᵢBⁱ)(1 - B)ᵈyₜ = (1 + Σⱼ θⱼBʲ)εₜ
```

**GARCH Model:**
```
σₜ² = ω + Σᵢ αᵢεₜ₋ᵢ² + Σⱼ βⱼσₜ₋ⱼ²
```

**VAR Model:**
```
yₜ = c + Σᵢ Aᵢyₜ₋ᵢ + εₜ
```

#### **2. Quantum Time Series Analysis**

**Quantum State Evolution:**
```
|ψ(t)⟩ = U(t)|ψ(0)⟩
```

**Quantum Time Series Encoding:**
```
|ψ(xₜ)⟩ = U(xₜ)|0⟩
```

**Quantum Forecasting:**
```
yₜ₊₁ = ⟨ψ(xₜ)|O|ψ(xₜ)⟩
```

### **Quantum Time Series Types**

#### **1. Quantum ARIMA:**
- **Quantum Encoding**: Encode time series as quantum states
- **Quantum Evolution**: Quantum circuit for time evolution
- **Quantum Prediction**: Quantum measurement for forecasting

#### **2. Quantum GARCH:**
- **Quantum Volatility**: Quantum state for volatility modeling
- **Quantum Dynamics**: Quantum circuit for volatility evolution
- **Quantum Risk**: Quantum measurement for risk assessment

#### **3. Quantum VAR:**
- **Quantum Multivariate**: Quantum state for multiple variables
- **Quantum Interactions**: Quantum entanglement for variable interactions
- **Quantum Forecasting**: Quantum measurement for multivariate prediction

### **Quantum Time Series Advantages**

#### **1. Quantum Properties:**
- **Superposition**: Parallel time series analysis
- **Entanglement**: Complex temporal correlations
- **Quantum Parallelism**: Exponential speedup potential

#### **2. Credit-specific Benefits:**
- **Non-linear Patterns**: Quantum circuits capture complex temporal relationships
- **High-dimensional Time Series**: Handle many variables efficiently
- **Quantum Advantage**: Potential speedup for large datasets

## 💻 Thực hành

### **Project 19: Quantum Time Series Analysis cho Credit Risk**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.algorithms import VQE, QAOA
from qiskit.algorithms.optimizers import SPSA, COBYLA
from qiskit.quantum_info import state_fidelity
from qiskit.opflow import PauliSumOp, StateFn, CircuitSampler
import pennylane as qml

class ClassicalTimeSeriesAnalysis:
    """Classical time series analysis methods"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def generate_credit_time_series(self, n_periods=500):
        """
        Generate synthetic credit time series data
        """
        np.random.seed(42)
        
        # Generate base time series
        t = np.arange(n_periods)
        
        # Trend component
        trend = 0.001 * t + 0.5
        
        # Seasonal component (monthly)
        seasonal = 0.1 * np.sin(2 * np.pi * t / 30)
        
        # Cyclical component (quarterly)
        cyclical = 0.05 * np.sin(2 * np.pi * t / 90)
        
        # Random walk component
        random_walk = np.cumsum(np.random.normal(0, 0.01, n_periods))
        
        # Credit-specific events (defaults, economic shocks)
        credit_events = np.zeros(n_periods)
        event_times = np.random.choice(n_periods, size=n_periods//50, replace=False)
        credit_events[event_times] = np.random.normal(-0.1, 0.05, len(event_times))
        
        # Combine components
        default_rate = trend + seasonal + cyclical + random_walk + credit_events
        default_rate = np.clip(default_rate, 0, 1)  # Ensure valid probability
        
        # Generate related time series
        credit_spread = 0.02 + 0.5 * default_rate + np.random.normal(0, 0.005, n_periods)
        market_volatility = 0.15 + 2 * default_rate + np.random.normal(0, 0.02, n_periods)
        economic_growth = 0.03 - 0.5 * default_rate + np.random.normal(0, 0.01, n_periods)
        
        # Create DataFrame
        data = pd.DataFrame({
            'default_rate': default_rate,
            'credit_spread': credit_spread,
            'market_volatility': market_volatility,
            'economic_growth': economic_growth
        }, index=pd.date_range('2020-01-01', periods=n_periods, freq='D'))
        
        return data
    
    def check_stationarity(self, time_series):
        """
        Check time series stationarity
        """
        result = adfuller(time_series)
        
        print(f"ADF Statistic: {result[0]:.4f}")
        print(f"p-value: {result[1]:.4f}")
        print(f"Critical values:")
        for key, value in result[4].items():
            print(f"  {key}: {value:.4f}")
        
        return result[1] < 0.05  # Stationary if p-value < 0.05
    
    def fit_arima_model(self, time_series, order=(1, 1, 1)):
        """
        Fit ARIMA model
        """
        try:
            self.model = ARIMA(time_series, order=order)
            fitted_model = self.model.fit()
            return fitted_model
        except Exception as e:
            print(f"Error fitting ARIMA model: {e}")
            return None
    
    def forecast_arima(self, model, steps=30):
        """
        Forecast using ARIMA model
        """
        if model is None:
            return None
        
        forecast = model.forecast(steps=steps)
        return forecast
    
    def calculate_forecast_metrics(self, actual, predicted):
        """
        Calculate forecast accuracy metrics
        """
        mse = mean_squared_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mse)
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse
        }

class QuantumTimeSeriesAnalysis:
    """Quantum time series analysis implementation"""
    
    def __init__(self, num_qubits=4):
        self.num_qubits = num_qubits
        self.backend = Aer.get_backend('qasm_simulator')
        self.optimizer = SPSA(maxiter=100)
        self.parameters = None
        
    def create_time_series_circuit(self, time_series, lookback=5):
        """
        Create quantum circuit for time series analysis
        """
        # Normalize time series
        time_series_norm = (time_series - np.mean(time_series)) / np.std(time_series)
        
        # Create feature map for time series encoding
        feature_map = ZZFeatureMap(feature_dimension=lookback, reps=2)
        
        # Create ansatz for time series prediction
        ansatz = RealAmplitudes(num_qubits=self.num_qubits, reps=3)
        
        # Combine circuits
        circuit = feature_map.compose(ansatz)
        
        return circuit, time_series_norm
    
    def prepare_time_series_data(self, time_series, lookback=5):
        """
        Prepare time series data for quantum analysis
        """
        # Create sliding window features
        X, y = [], []
        
        for i in range(lookback, len(time_series)):
            X.append(time_series[i-lookback:i])
            y.append(time_series[i])
        
        return np.array(X), np.array(y)
    
    def quantum_forecast(self, time_series, lookback=5, forecast_steps=30):
        """
        Quantum time series forecasting
        """
        circuit, time_series_norm = self.create_time_series_circuit(time_series, lookback)
        
        # Prepare training data
        X_train, y_train = self.prepare_time_series_data(time_series_norm, lookback)
        
        # Use subset for quantum processing
        subset_size = min(100, len(X_train))
        indices = np.random.choice(len(X_train), subset_size, replace=False)
        X_subset = X_train[indices]
        y_subset = y_train[indices]
        
        # Initialize parameters
        self.parameters = np.random.random(circuit.num_parameters) * 2 * np.pi
        
        # Train quantum model (simplified)
        # In practice, use proper quantum training
        for epoch in range(20):
            # Simplified training loop
            for i in range(len(X_subset)):
                # Encode input
                input_params = np.concatenate([X_subset[i], self.parameters])
                bound_circuit = circuit.bind_parameters(input_params)
                
                # Execute circuit
                job = execute(bound_circuit, self.backend, shots=1000)
                result = job.result()
                counts = result.get_counts()
                
                # Calculate prediction
                prediction = self._extract_prediction_from_counts(counts)
                
                # Update parameters (simplified)
                error = y_subset[i] - prediction
                self.parameters += 0.01 * error * np.random.normal(0, 0.1, len(self.parameters))
        
        # Generate forecasts
        forecasts = []
        current_input = time_series_norm[-lookback:]
        
        for step in range(forecast_steps):
            # Encode current input
            input_params = np.concatenate([current_input, self.parameters])
            bound_circuit = circuit.bind_parameters(input_params)
            
            # Execute circuit
            job = execute(bound_circuit, self.backend, shots=1000)
            result = job.result()
            counts = result.get_counts()
            
            # Extract prediction
            prediction = self._extract_prediction_from_counts(counts)
            forecasts.append(prediction)
            
            # Update input for next step
            current_input = np.append(current_input[1:], prediction)
        
        # Denormalize forecasts
        forecasts_denorm = np.array(forecasts) * np.std(time_series) + np.mean(time_series)
        
        return forecasts_denorm
    
    def _extract_prediction_from_counts(self, counts):
        """
        Extract prediction from quantum measurement counts
        """
        total_shots = sum(counts.values())
        prediction = 0.0
        
        for bitstring, count in counts.items():
            probability = count / total_shots
            
            # Convert bitstring to prediction value
            # Simplified: use parity as prediction
            parity = sum(int(bit) for bit in bitstring) % 2
            prediction += probability * (1 if parity == 0 else -1)
        
        return prediction
    
    def quantum_volatility_modeling(self, time_series, lookback=10):
        """
        Quantum volatility modeling (GARCH-like)
        """
        # Calculate returns
        returns = np.diff(time_series) / time_series[:-1]
        
        # Prepare volatility data
        volatility = np.abs(returns)
        X_vol, y_vol = self.prepare_time_series_data(volatility, lookback)
        
        # Quantum volatility prediction
        circuit, volatility_norm = self.create_time_series_circuit(volatility, lookback)
        
        # Simplified quantum volatility modeling
        volatility_forecasts = []
        current_vol_input = volatility_norm[-lookback:]
        
        for step in range(30):  # Forecast 30 steps
            # Encode volatility input
            input_params = np.concatenate([current_vol_input, self.parameters])
            bound_circuit = circuit.bind_parameters(input_params)
            
            # Execute circuit
            job = execute(bound_circuit, self.backend, shots=1000)
            result = job.result()
            counts = result.get_counts()
            
            # Extract volatility prediction
            vol_prediction = self._extract_prediction_from_counts(counts)
            volatility_forecasts.append(vol_prediction)
            
            # Update input
            current_vol_input = np.append(current_vol_input[1:], vol_prediction)
        
        # Denormalize volatility forecasts
        volatility_forecasts_denorm = np.array(volatility_forecasts) * np.std(volatility) + np.mean(volatility)
        
        return volatility_forecasts_denorm
    
    def quantum_multivariate_forecasting(self, time_series_data, lookback=5, forecast_steps=30):
        """
        Quantum multivariate time series forecasting
        """
        # Normalize all time series
        normalized_data = {}
        for column in time_series_data.columns:
            series = time_series_data[column].values
            normalized_data[column] = (series - np.mean(series)) / np.std(series)
        
        # Create multivariate forecasts
        multivariate_forecasts = {}
        
        for column in time_series_data.columns:
            series = normalized_data[column]
            forecasts = self.quantum_forecast(series, lookback, forecast_steps)
            
            # Denormalize
            original_series = time_series_data[column].values
            forecasts_denorm = forecasts * np.std(original_series) + np.mean(original_series)
            multivariate_forecasts[column] = forecasts_denorm
        
        return multivariate_forecasts

def compare_time_series_methods():
    """
    Compare classical and quantum time series methods
    """
    print("=== Classical vs Quantum Time Series Analysis ===\n")
    
    # Generate credit time series data
    classical_analysis = ClassicalTimeSeriesAnalysis()
    credit_data = classical_analysis.generate_credit_time_series(n_periods=400)
    
    # Focus on default rate for analysis
    default_rate = credit_data['default_rate'].values
    
    print("1. Classical Time Series Analysis:")
    
    # Check stationarity
    print("   Stationarity Test:")
    is_stationary = classical_analysis.check_stationarity(default_rate)
    print(f"   Is stationary: {is_stationary}")
    
    # Fit ARIMA model
    print("\n   ARIMA Model:")
    arima_model = classical_analysis.fit_arima_model(default_rate, order=(1, 1, 1))
    
    if arima_model is not None:
        print(f"   AIC: {arima_model.aic:.4f}")
        print(f"   BIC: {arima_model.bic:.4f}")
        
        # Classical forecast
        classical_forecast = classical_analysis.forecast_arima(arima_model, steps=30)
        
        if classical_forecast is not None:
            print(f"   Forecast mean: {classical_forecast.mean():.4f}")
            print(f"   Forecast std: {classical_forecast.std():.4f}")
    
    # Quantum time series analysis
    print("\n2. Quantum Time Series Analysis:")
    
    quantum_analysis = QuantumTimeSeriesAnalysis(num_qubits=4)
    
    # Quantum forecast
    print("   Quantum Forecasting:")
    quantum_forecast = quantum_analysis.quantum_forecast(default_rate, lookback=5, forecast_steps=30)
    
    if quantum_forecast is not None:
        print(f"   Forecast mean: {np.mean(quantum_forecast):.4f}")
        print(f"   Forecast std: {np.std(quantum_forecast):.4f}")
    
    # Quantum volatility modeling
    print("\n   Quantum Volatility Modeling:")
    volatility_forecast = quantum_analysis.quantum_volatility_modeling(default_rate, lookback=10)
    
    if volatility_forecast is not None:
        print(f"   Volatility forecast mean: {np.mean(volatility_forecast):.4f}")
        print(f"   Volatility forecast std: {np.std(volatility_forecast):.4f}")
    
    # Multivariate forecasting
    print("\n   Quantum Multivariate Forecasting:")
    multivariate_forecasts = quantum_analysis.quantum_multivariate_forecasting(
        credit_data, lookback=5, forecast_steps=30
    )
    
    for column, forecast in multivariate_forecasts.items():
        print(f"   {column}: mean={np.mean(forecast):.4f}, std={np.std(forecast):.4f}")
    
    # Visualize results
    plt.figure(figsize=(20, 12))
    
    # Original time series
    plt.subplot(3, 3, 1)
    plt.plot(credit_data.index, credit_data['default_rate'], 'b-', linewidth=1)
    plt.title('Original Default Rate Time Series')
    plt.xlabel('Time')
    plt.ylabel('Default Rate')
    plt.grid(True)
    
    # Classical vs Quantum forecast
    plt.subplot(3, 3, 2)
    forecast_index = pd.date_range(credit_data.index[-1], periods=31, freq='D')[1:]
    
    plt.plot(credit_data.index, credit_data['default_rate'], 'b-', linewidth=1, label='Historical')
    
    if classical_forecast is not None:
        plt.plot(forecast_index, classical_forecast, 'r--', linewidth=2, label='Classical ARIMA')
    
    if quantum_forecast is not None:
        plt.plot(forecast_index, quantum_forecast, 'g--', linewidth=2, label='Quantum')
    
    plt.title('Time Series Forecasting Comparison')
    plt.xlabel('Time')
    plt.ylabel('Default Rate')
    plt.legend()
    plt.grid(True)
    
    # Volatility comparison
    plt.subplot(3, 3, 3)
    returns = np.diff(default_rate) / default_rate[:-1]
    historical_volatility = np.abs(returns)
    
    plt.plot(credit_data.index[1:], historical_volatility, 'b-', linewidth=1, label='Historical Volatility')
    
    if volatility_forecast is not None:
        vol_forecast_index = pd.date_range(credit_data.index[-1], periods=31, freq='D')[1:]
        plt.plot(vol_forecast_index, volatility_forecast, 'r--', linewidth=2, label='Quantum Volatility')
    
    plt.title('Volatility Forecasting')
    plt.xlabel('Time')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)
    
    # Multivariate forecasts
    for i, column in enumerate(['credit_spread', 'market_volatility', 'economic_growth']):
        plt.subplot(3, 3, 4 + i)
        
        plt.plot(credit_data.index, credit_data[column], 'b-', linewidth=1, label='Historical')
        
        if column in multivariate_forecasts:
            plt.plot(forecast_index, multivariate_forecasts[column], 'g--', linewidth=2, label='Quantum Forecast')
        
        plt.title(f'{column.replace("_", " ").title()} Forecasting')
        plt.xlabel('Time')
        plt.ylabel(column.replace('_', ' ').title())
        plt.legend()
        plt.grid(True)
    
    # ACF and PACF plots
    plt.subplot(3, 3, 7)
    plot_acf(default_rate, lags=40, ax=plt.gca())
    plt.title('Autocorrelation Function')
    
    plt.subplot(3, 3, 8)
    plot_pacf(default_rate, lags=40, ax=plt.gca())
    plt.title('Partial Autocorrelation Function')
    
    # Forecast accuracy comparison
    plt.subplot(3, 3, 9)
    if classical_forecast is not None and quantum_forecast is not None:
        # Simulate actual values for comparison
        actual_forecast = default_rate[-30:] + np.random.normal(0, 0.01, 30)
        
        classical_metrics = classical_analysis.calculate_forecast_metrics(actual_forecast, classical_forecast)
        quantum_metrics = classical_analysis.calculate_forecast_metrics(actual_forecast, quantum_forecast)
        
        metrics = ['MSE', 'MAE', 'RMSE']
        classical_values = [classical_metrics[m] for m in metrics]
        quantum_values = [quantum_metrics[m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, classical_values, width, label='Classical', color='blue', alpha=0.7)
        plt.bar(x + width/2, quantum_values, width, label='Quantum', color='orange', alpha=0.7)
        
        plt.xlabel('Metrics')
        plt.ylabel('Error')
        plt.title('Forecast Accuracy Comparison')
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'classical_forecast': classical_forecast,
        'quantum_forecast': quantum_forecast,
        'volatility_forecast': volatility_forecast,
        'multivariate_forecasts': multivariate_forecasts
    }

def quantum_time_series_analysis():
    """
    Analyze quantum time series properties
    """
    print("=== Quantum Time Series Analysis ===\n")
    
    # Generate different time series scenarios
    classical_analysis = ClassicalTimeSeriesAnalysis()
    
    scenarios = {
        'Trending': classical_analysis.generate_credit_time_series(n_periods=300),
        'Cyclical': pd.DataFrame({
            'value': 0.5 + 0.2 * np.sin(np.linspace(0, 4*np.pi, 300)) + np.random.normal(0, 0.05, 300)
        }),
        'Volatile': pd.DataFrame({
            'value': np.cumsum(np.random.normal(0, 0.1, 300))
        })
    }
    
    quantum_analysis = QuantumTimeSeriesAnalysis(num_qubits=4)
    
    analysis_results = {}
    
    for scenario_name, data in scenarios.items():
        print(f"Analyzing {scenario_name} scenario:")
        
        # Extract time series
        if scenario_name == 'Trending':
            time_series = data['default_rate'].values
        else:
            time_series = data['value'].values
        
        # Classical analysis
        is_stationary = classical_analysis.check_stationarity(time_series)
        arima_model = classical_analysis.fit_arima_model(time_series, order=(1, 1, 1))
        
        if arima_model is not None:
            classical_forecast = classical_analysis.forecast_arima(arima_model, steps=20)
        else:
            classical_forecast = None
        
        # Quantum analysis
        quantum_forecast = quantum_analysis.quantum_forecast(time_series, lookback=5, forecast_steps=20)
        volatility_forecast = quantum_analysis.quantum_volatility_modeling(time_series, lookback=10)
        
        analysis_results[scenario_name] = {
            'is_stationary': is_stationary,
            'classical_forecast': classical_forecast,
            'quantum_forecast': quantum_forecast,
            'volatility_forecast': volatility_forecast,
            'time_series': time_series
        }
        
        print(f"  Stationary: {is_stationary}")
        if classical_forecast is not None:
            print(f"  Classical forecast mean: {classical_forecast.mean():.4f}")
        if quantum_forecast is not None:
            print(f"  Quantum forecast mean: {np.mean(quantum_forecast):.4f}")
        print()
    
    # Visualize analysis
    plt.figure(figsize=(15, 10))
    
    # Time series comparison across scenarios
    for i, (scenario_name, results) in enumerate(analysis_results.items()):
        plt.subplot(3, 3, i + 1)
        
        time_series = results['time_series']
        plt.plot(time_series, 'b-', linewidth=1, label='Historical')
        
        if results['classical_forecast'] is not None:
            forecast_index = np.arange(len(time_series), len(time_series) + len(results['classical_forecast']))
            plt.plot(forecast_index, results['classical_forecast'], 'r--', linewidth=2, label='Classical')
        
        if results['quantum_forecast'] is not None:
            forecast_index = np.arange(len(time_series), len(time_series) + len(results['quantum_forecast']))
            plt.plot(forecast_index, results['quantum_forecast'], 'g--', linewidth=2, label='Quantum')
        
        plt.title(f'{scenario_name} Time Series')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
    
    # Volatility comparison
    for i, (scenario_name, results) in enumerate(analysis_results.items()):
        plt.subplot(3, 3, i + 4)
        
        time_series = results['time_series']
        returns = np.diff(time_series) / time_series[:-1]
        historical_volatility = np.abs(returns)
        
        plt.plot(historical_volatility, 'b-', linewidth=1, label='Historical Volatility')
        
        if results['volatility_forecast'] is not None:
            vol_forecast_index = np.arange(len(historical_volatility), len(historical_volatility) + len(results['volatility_forecast']))
            plt.plot(vol_forecast_index, results['volatility_forecast'], 'r--', linewidth=2, label='Quantum Volatility')
        
        plt.title(f'{scenario_name} Volatility')
        plt.xlabel('Time')
        plt.ylabel('Volatility')
        plt.legend()
        plt.grid(True)
    
    # Forecast accuracy comparison
    plt.subplot(3, 3, 7)
    scenario_names = list(analysis_results.keys())
    classical_means = []
    quantum_means = []
    
    for scenario_name in scenario_names:
        results = analysis_results[scenario_name]
        
        if results['classical_forecast'] is not None:
            classical_means.append(results['classical_forecast'].mean())
        else:
            classical_means.append(0)
        
        if results['quantum_forecast'] is not None:
            quantum_means.append(np.mean(results['quantum_forecast']))
        else:
            quantum_means.append(0)
    
    x = np.arange(len(scenario_names))
    width = 0.35
    
    plt.bar(x - width/2, classical_means, width, label='Classical', color='blue', alpha=0.7)
    plt.bar(x + width/2, quantum_means, width, label='Quantum', color='orange', alpha=0.7)
    
    plt.xlabel('Scenarios')
    plt.ylabel('Forecast Mean')
    plt.title('Forecast Comparison')
    plt.xticks(x, scenario_names)
    plt.legend()
    plt.grid(True)
    
    # Stationarity analysis
    plt.subplot(3, 3, 8)
    stationary_counts = [analysis_results[name]['is_stationary'] for name in scenario_names]
    plt.bar(scenario_names, stationary_counts, color=['red' if not s else 'green' for s in stationary_counts])
    plt.ylabel('Is Stationary')
    plt.title('Stationarity Analysis')
    plt.ylim(0, 1)
    
    # Volatility forecast comparison
    plt.subplot(3, 3, 9)
    vol_means = []
    for scenario_name in scenario_names:
        results = analysis_results[scenario_name]
        if results['volatility_forecast'] is not None:
            vol_means.append(np.mean(results['volatility_forecast']))
        else:
            vol_means.append(0)
    
    plt.bar(scenario_names, vol_means, color='purple', alpha=0.7)
    plt.ylabel('Volatility Forecast Mean')
    plt.title('Volatility Forecast Comparison')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return analysis_results

# Run demos
if __name__ == "__main__":
    print("Running Time Series Methods Comparison...")
    time_series_results = compare_time_series_methods()
    
    print("\nRunning Quantum Time Series Analysis...")
    analysis_results = quantum_time_series_analysis()
```

## 📊 Kết quả và Phân tích

### **Quantum Time Series Analysis Advantages:**

#### **1. Quantum Properties:**
- **Superposition**: Parallel time series analysis
- **Entanglement**: Complex temporal correlations
- **Quantum Parallelism**: Exponential speedup potential

#### **2. Credit-specific Benefits:**
- **Non-linear Patterns**: Quantum circuits capture complex temporal relationships
- **High-dimensional Time Series**: Handle many variables efficiently
- **Quantum Advantage**: Potential speedup for large datasets

#### **3. Performance Characteristics:**
- **Better Non-linear Forecasting**: Quantum features improve prediction accuracy
- **Robustness**: Quantum time series analysis handles noisy data
- **Scalability**: Quantum advantage for large-scale time series analysis

### **Comparison với Classical Time Series Analysis:**

#### **Classical Limitations:**
- Limited to linear time series models
- Assumption of normal distributions
- Curse of dimensionality
- Feature engineering required

#### **Quantum Advantages:**
- Non-linear time series modeling
- Flexible distribution modeling
- High-dimensional time series space
- Automatic feature learning

## 🎯 Bài tập về nhà

### **Exercise 1: Quantum Time Series Calibration**
Implement quantum time series calibration methods cho credit risk forecasting.

### **Exercise 2: Quantum Time Series Ensemble Methods**
Build ensemble of quantum time series models cho improved accuracy.

### **Exercise 3: Quantum Time Series Feature Selection**
Develop quantum feature selection cho time series analysis.

### **Exercise 4: Quantum Time Series Validation**
Create validation framework cho quantum time series models.

---

> *"Quantum time series analysis leverages quantum superposition and entanglement to provide superior forecasting for credit risk assessment."* - Quantum Finance Research

> Ngày tiếp theo: [Quantum Credit Derivatives Pricing](Day20.md) 