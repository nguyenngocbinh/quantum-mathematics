# Day 32: Quantum Random Number Generation

## 🎯 Mục tiêu
- Hiểu nguyên lý tạo số ngẫu nhiên lượng tử
- Triển khai các phương pháp QRNG khác nhau
- Đánh giá chất lượng và entropy của số ngẫu nhiên
- Ứng dụng QRNG trong cryptography và simulation

## 🎲 Quantum Random Number Generation - Tổng Quan

### Tại sao Quantum Random Number Generation?
- **True randomness**: Dựa trên nguyên lý bất định của cơ học lượng tử
- **Unpredictability**: Không thể dự đoán kết quả trước khi đo
- **Cryptographic security**: Entropy cao cho ứng dụng bảo mật
- **Simulation accuracy**: Cần thiết cho Monte Carlo và các phương pháp simulation
- **Gaming and gambling**: Fairness và unpredictability

```python
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import random_statevector, Statevector
from qiskit.visualization import plot_bloch_multivector, plot_histogram
import numpy as np
import matplotlib.pyplot as plt
import random
import hashlib
from typing import List, Tuple, Dict
import seaborn as sns
from scipy import stats
import time
```

## 🔧 Quantum Random Number Generation Methods

### 1. Measurement-based QRNG

```python
class MeasurementBasedQRNG:
    """
    QRNG dựa trên đo lường trạng thái lượng tử
    """
    
    def __init__(self, backend='qasm_simulator'):
        self.backend = Aer.get_backend(backend)
        
    def generate_random_bits(self, n_bits: int, method: str = 'hadamard') -> List[int]:
        """
        Tạo n_bits ngẫu nhiên bằng phương pháp đo lường
        """
        if method == 'hadamard':
            return self._hadamard_method(n_bits)
        elif method == 'rotation':
            return self._rotation_method(n_bits)
        elif method == 'bell':
            return self._bell_method(n_bits)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _hadamard_method(self, n_bits: int) -> List[int]:
        """
        Sử dụng Hadamard gate để tạo superposition
        """
        # Tạo circuit với n_bits qubits
        qc = QuantumCircuit(n_bits, n_bits)
        
        # Áp dụng Hadamard gate cho mỗi qubit
        for i in range(n_bits):
            qc.h(i)
        
        # Đo tất cả qubits
        qc.measure_all()
        
        # Thực hiện đo
        job = execute(qc, self.backend, shots=1)
        result = job.result()
        counts = result.get_counts()
        
        # Lấy kết quả
        bit_string = list(counts.keys())[0]
        return [int(bit) for bit in bit_string]
    
    def _rotation_method(self, n_bits: int) -> List[int]:
        """
        Sử dụng random rotation để tạo randomness
        """
        qc = QuantumCircuit(n_bits, n_bits)
        
        for i in range(n_bits):
            # Random rotation angles
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, 2*np.pi)
            
            qc.rx(theta, i)
            qc.rz(phi, i)
        
        qc.measure_all()
        
        job = execute(qc, self.backend, shots=1)
        result = job.result()
        counts = result.get_counts()
        
        bit_string = list(counts.keys())[0]
        return [int(bit) for bit in bit_string]
    
    def _bell_method(self, n_bits: int) -> List[int]:
        """
        Sử dụng Bell states để tạo randomness
        """
        # Tạo n_bits/2 Bell pairs
        n_pairs = n_bits // 2
        qc = QuantumCircuit(2 * n_pairs, 2 * n_pairs)
        
        for i in range(n_pairs):
            # Tạo Bell state
            qc.h(2*i)
            qc.cx(2*i, 2*i + 1)
        
        qc.measure_all()
        
        job = execute(qc, self.backend, shots=1)
        result = job.result()
        counts = result.get_counts()
        
        bit_string = list(counts.keys())[0]
        return [int(bit) for bit in bit_string]

# Test Measurement-based QRNG
qrng = MeasurementBasedQRNG()

print("Measurement-based QRNG Demo:")
print("Hadamard method:")
hadamard_bits = qrng.generate_random_bits(8, 'hadamard')
print(f"Generated bits: {hadamard_bits}")

print("\nRotation method:")
rotation_bits = qrng.generate_random_bits(8, 'rotation')
print(f"Generated bits: {rotation_bits}")

print("\nBell method:")
bell_bits = qrng.generate_random_bits(8, 'bell')
print(f"Generated bits: {bell_bits}")
```

### 2. Entanglement-based QRNG

```python
class EntanglementBasedQRNG:
    """
    QRNG dựa trên quantum entanglement
    """
    
    def __init__(self, backend='qasm_simulator'):
        self.backend = Aer.get_backend(backend)
    
    def generate_from_bell_state(self, n_pairs: int) -> Tuple[List[int], List[int]]:
        """
        Tạo randomness từ Bell states
        """
        qc = QuantumCircuit(2 * n_pairs, 2 * n_pairs)
        
        # Tạo Bell pairs
        for i in range(n_pairs):
            qc.h(2*i)
            qc.cx(2*i, 2*i + 1)
        
        # Đo với basis ngẫu nhiên
        alice_bits = []
        bob_bits = []
        
        for i in range(n_pairs):
            # Alice measures first qubit
            alice_qc = qc.copy()
            alice_qc.measure(2*i, 2*i)
            
            # Bob measures second qubit
            bob_qc = qc.copy()
            bob_qc.measure(2*i + 1, 2*i + 1)
            
            # Execute measurements
            alice_job = execute(alice_qc, self.backend, shots=1)
            alice_result = alice_job.result()
            alice_bit = int(list(alice_result.get_counts().keys())[0][0])
            
            bob_job = execute(bob_qc, self.backend, shots=1)
            bob_result = bob_job.result()
            bob_bit = int(list(bob_result.get_counts().keys())[0][1])
            
            alice_bits.append(alice_bit)
            bob_bits.append(bob_bit)
        
        return alice_bits, bob_bits
    
    def generate_from_ghz_state(self, n_qubits: int) -> List[int]:
        """
        Tạo randomness từ GHZ state
        """
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        # Tạo GHZ state
        qc.h(0)
        for i in range(1, n_qubits):
            qc.cx(0, i)
        
        qc.measure_all()
        
        job = execute(qc, self.backend, shots=1)
        result = job.result()
        counts = result.get_counts()
        
        bit_string = list(counts.keys())[0]
        return [int(bit) for bit in bit_string]

# Test Entanglement-based QRNG
ent_qrng = EntanglementBasedQRNG()

print("\nEntanglement-based QRNG Demo:")
print("Bell state method:")
alice_bits, bob_bits = ent_qrng.generate_from_bell_state(4)
print(f"Alice's bits: {alice_bits}")
print(f"Bob's bits: {bob_bits}")

print("\nGHZ state method:")
ghz_bits = ent_qrng.generate_from_ghz_state(5)
print(f"GHZ bits: {ghz_bits}")
```

## 📊 Randomness Quality Assessment

### 1. Statistical Tests

```python
class RandomnessTester:
    """
    Đánh giá chất lượng số ngẫu nhiên
    """
    
    def __init__(self):
        self.tests = {
            'frequency': self._frequency_test,
            'runs': self._runs_test,
            'autocorrelation': self._autocorrelation_test,
            'entropy': self._entropy_test,
            'chi_square': self._chi_square_test
        }
    
    def run_all_tests(self, bit_sequence: List[int]) -> Dict:
        """
        Chạy tất cả các test thống kê
        """
        results = {}
        
        for test_name, test_func in self.tests.items():
            try:
                results[test_name] = test_func(bit_sequence)
            except Exception as e:
                results[test_name] = f"Error: {str(e)}"
        
        return results
    
    def _frequency_test(self, bits: List[int]) -> Dict:
        """
        Test tần suất 0 và 1
        """
        n = len(bits)
        ones = sum(bits)
        zeros = n - ones
        
        proportion = ones / n
        expected = 0.5
        
        # Chi-square test
        chi_square = ((ones - n/2)**2 + (zeros - n/2)**2) / (n/2)
        p_value = 1 - stats.chi2.cdf(chi_square, 1)
        
        return {
            'proportion_ones': proportion,
            'chi_square': chi_square,
            'p_value': p_value,
            'pass': p_value > 0.01
        }
    
    def _runs_test(self, bits: List[int]) -> Dict:
        """
        Test runs (chuỗi liên tiếp)
        """
        n = len(bits)
        ones = sum(bits)
        zeros = n - ones
        
        # Count runs
        runs = 1
        for i in range(1, n):
            if bits[i] != bits[i-1]:
                runs += 1
        
        # Expected number of runs
        expected_runs = 1 + 2 * ones * zeros / n
        variance = 2 * ones * zeros * (2 * ones * zeros - n) / (n**2 * (n-1))
        
        # Z-score
        z_score = (runs - expected_runs) / np.sqrt(variance)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return {
            'runs': runs,
            'expected_runs': expected_runs,
            'z_score': z_score,
            'p_value': p_value,
            'pass': p_value > 0.01
        }
    
    def _autocorrelation_test(self, bits: List[int], lag: int = 1) -> Dict:
        """
        Test autocorrelation
        """
        n = len(bits)
        if lag >= n:
            return {'error': 'Lag too large'}
        
        # Calculate autocorrelation
        autocorr = 0
        for i in range(n - lag):
            autocorr += bits[i] * bits[i + lag]
        
        autocorr = autocorr / (n - lag)
        
        # Expected value for random sequence
        expected = 0.25  # E[X*X] = E[X]^2 = 0.25
        
        return {
            'autocorrelation': autocorr,
            'expected': expected,
            'deviation': abs(autocorr - expected),
            'pass': abs(autocorr - expected) < 0.1
        }
    
    def _entropy_test(self, bits: List[int]) -> Dict:
        """
        Test entropy
        """
        n = len(bits)
        ones = sum(bits)
        zeros = n - ones
        
        # Calculate Shannon entropy
        p1 = ones / n
        p0 = zeros / n
        
        if p1 == 0 or p0 == 0:
            entropy = 0
        else:
            entropy = -p1 * np.log2(p1) - p0 * np.log2(p0)
        
        return {
            'entropy': entropy,
            'max_entropy': 1.0,
            'entropy_ratio': entropy,
            'pass': entropy > 0.9
        }
    
    def _chi_square_test(self, bits: List[int]) -> Dict:
        """
        Chi-square test cho distribution
        """
        n = len(bits)
        ones = sum(bits)
        zeros = n - ones
        
        # Chi-square statistic
        chi_square = ((ones - n/2)**2 + (zeros - n/2)**2) / (n/2)
        p_value = 1 - stats.chi2.cdf(chi_square, 1)
        
        return {
            'chi_square': chi_square,
            'p_value': p_value,
            'pass': p_value > 0.01
        }

# Test randomness quality
tester = RandomnessTester()

print("\nRandomness Quality Assessment:")
# Generate test sequence
test_bits = qrng.generate_random_bits(1000, 'hadamard')
test_results = tester.run_all_tests(test_bits)

for test_name, result in test_results.items():
    print(f"\n{test_name.upper()} Test:")
    if isinstance(result, dict):
        for key, value in result.items():
            print(f"  {key}: {value}")
    else:
        print(f"  Result: {result}")
```

### 2. Entropy Analysis

```python
def analyze_entropy_sources():
    """
    Phân tích entropy từ các nguồn khác nhau
    """
    sources = {
        'Quantum Hadamard': qrng.generate_random_bits(1000, 'hadamard'),
        'Quantum Rotation': qrng.generate_random_bits(1000, 'rotation'),
        'Quantum Bell': qrng.generate_random_bits(1000, 'bell'),
        'Classical Python': [random.randint(0, 1) for _ in range(1000)],
        'Pseudo-random': np.random.randint(0, 2, 1000).tolist()
    }
    
    entropy_analysis = {}
    
    for source_name, bits in sources.items():
        # Calculate entropy
        ones = sum(bits)
        zeros = len(bits) - ones
        p1 = ones / len(bits)
        p0 = zeros / len(bits)
        
        if p1 == 0 or p0 == 0:
            entropy = 0
        else:
            entropy = -p1 * np.log2(p1) - p0 * np.log2(p0)
        
        # Run statistical tests
        test_results = tester.run_all_tests(bits)
        passed_tests = sum(1 for result in test_results.values() 
                          if isinstance(result, dict) and result.get('pass', False))
        
        entropy_analysis[source_name] = {
            'entropy': entropy,
            'passed_tests': passed_tests,
            'total_tests': len(test_results)
        }
    
    return entropy_analysis

# Analyze entropy sources
print("\nEntropy Source Analysis:")
entropy_results = analyze_entropy_sources()

for source, data in entropy_results.items():
    print(f"\n{source}:")
    print(f"  Entropy: {data['entropy']:.4f}")
    print(f"  Tests passed: {data['passed_tests']}/{data['total_tests']}")
```

## 🔐 Cryptographic Applications

### 1. Quantum Key Generation

```python
def quantum_key_generation(n_bits: int = 256) -> Tuple[List[int], List[int]]:
    """
    Tạo cryptographic key sử dụng QRNG
    """
    # Generate random bits
    alice_bits = qrng.generate_random_bits(n_bits, 'hadamard')
    bob_bits = qrng.generate_random_bits(n_bits, 'hadamard')
    
    # Test quality
    alice_quality = tester.run_all_tests(alice_bits)
    bob_quality = tester.run_all_tests(bob_bits)
    
    # Check if keys are suitable for cryptography
    alice_entropy = alice_quality['entropy']['entropy']
    bob_entropy = bob_quality['entropy']['entropy']
    
    if alice_entropy < 0.95 or bob_entropy < 0.95:
        print("Warning: Low entropy detected in key generation")
    
    return alice_bits, bob_bits

def quantum_otp_encryption(message: str, key: List[int]) -> List[int]:
    """
    One-time pad encryption sử dụng quantum key
    """
    # Convert message to binary
    message_bits = []
    for char in message:
        char_bits = [int(bit) for bit in format(ord(char), '08b')]
        message_bits.extend(char_bits)
    
    # Pad message if necessary
    while len(message_bits) < len(key):
        message_bits.extend([0] * 8)  # Pad with null bytes
    
    # Truncate key if necessary
    key = key[:len(message_bits)]
    
    # XOR encryption
    cipher_bits = [m ^ k for m, k in zip(message_bits, key)]
    
    return cipher_bits

def quantum_otp_decryption(cipher_bits: List[int], key: List[int]) -> str:
    """
    One-time pad decryption
    """
    # XOR decryption
    message_bits = [c ^ k for c, k in zip(cipher_bits, key)]
    
    # Convert back to string
    message = ""
    for i in range(0, len(message_bits), 8):
        byte_bits = message_bits[i:i+8]
        if len(byte_bits) == 8:
            char_code = int(''.join(map(str, byte_bits)), 2)
            if char_code != 0:  # Skip null bytes
                message += chr(char_code)
    
    return message

# Test quantum cryptography
print("\nQuantum Cryptography Demo:")
# Generate quantum key
alice_key, bob_key = quantum_key_generation(256)

# Test OTP encryption
message = "Hello Quantum World!"
print(f"Original message: {message}")

# Encrypt with Alice's key
cipher = quantum_otp_encryption(message, alice_key)
print(f"Cipher length: {len(cipher)} bits")

# Decrypt with Bob's key (should be different)
decrypted = quantum_otp_decryption(cipher, bob_key)
print(f"Decrypted with Bob's key: {decrypted}")

# Decrypt with Alice's key (should be correct)
correct_decrypted = quantum_otp_decryption(cipher, alice_key)
print(f"Decrypted with Alice's key: {correct_decrypted}")
```

### 2. Quantum Random Number Service

```python
class QuantumRandomService:
    """
    Service cung cấp số ngẫu nhiên lượng tử
    """
    
    def __init__(self):
        self.qrng = MeasurementBasedQRNG()
        self.cache = []
        self.cache_size = 1000
    
    def get_random_bits(self, n_bits: int) -> List[int]:
        """
        Lấy n_bits ngẫu nhiên
        """
        if len(self.cache) < n_bits:
            # Generate more bits
            new_bits = self.qrng.generate_random_bits(self.cache_size, 'hadamard')
            self.cache.extend(new_bits)
        
        # Return requested bits
        requested_bits = self.cache[:n_bits]
        self.cache = self.cache[n_bits:]
        
        return requested_bits
    
    def get_random_int(self, min_val: int, max_val: int) -> int:
        """
        Lấy số nguyên ngẫu nhiên trong khoảng [min_val, max_val]
        """
        range_size = max_val - min_val + 1
        n_bits = int(np.ceil(np.log2(range_size)))
        
        while True:
            bits = self.get_random_bits(n_bits)
            value = int(''.join(map(str, bits)), 2)
            
            if value < range_size:
                return min_val + value
    
    def get_random_float(self) -> float:
        """
        Lấy số thực ngẫu nhiên trong [0, 1)
        """
        bits = self.get_random_bits(32)
        value = int(''.join(map(str, bits)), 2)
        return value / (2**32)
    
    def get_random_string(self, length: int, charset: str = "0123456789ABCDEF") -> str:
        """
        Lấy chuỗi ngẫu nhiên
        """
        result = ""
        charset_size = len(charset)
        
        for _ in range(length):
            index = self.get_random_int(0, charset_size - 1)
            result += charset[index]
        
        return result

# Test quantum random service
print("\nQuantum Random Service Demo:")
qrng_service = QuantumRandomService()

print(f"Random bits: {qrng_service.get_random_bits(16)}")
print(f"Random int (1-100): {qrng_service.get_random_int(1, 100)}")
print(f"Random float: {qrng_service.get_random_float():.6f}")
print(f"Random string: {qrng_service.get_random_string(10)}")
```

## 🎲 Gaming and Simulation Applications

### 1. Quantum Dice

```python
def quantum_dice_roll(sides: int = 6) -> int:
    """
    Gieo xúc xắc lượng tử
    """
    qrng_service = QuantumRandomService()
    return qrng_service.get_random_int(1, sides)

def quantum_card_shuffle(deck_size: int = 52) -> List[int]:
    """
    Xáo bài lượng tử
    """
    qrng_service = QuantumRandomService()
    deck = list(range(deck_size))
    shuffled = []
    
    while deck:
        index = qrng_service.get_random_int(0, len(deck) - 1)
        shuffled.append(deck.pop(index))
    
    return shuffled

# Test quantum gaming
print("\nQuantum Gaming Demo:")
print("Quantum dice rolls:")
for _ in range(10):
    roll = quantum_dice_roll(6)
    print(f"Roll: {roll}")

print("\nQuantum card shuffle (first 10 cards):")
shuffled_deck = quantum_card_shuffle(52)
print(f"First 10 cards: {shuffled_deck[:10]}")
```

### 2. Monte Carlo Simulation

```python
def quantum_monte_carlo_pi(n_points: int = 10000) -> float:
    """
    Tính π sử dụng Monte Carlo với QRNG
    """
    qrng_service = QuantumRandomService()
    inside_circle = 0
    
    for _ in range(n_points):
        x = qrng_service.get_random_float() * 2 - 1  # [-1, 1]
        y = qrng_service.get_random_float() * 2 - 1  # [-1, 1]
        
        if x**2 + y**2 <= 1:
            inside_circle += 1
    
    pi_estimate = 4 * inside_circle / n_points
    return pi_estimate

def quantum_integration(func, a: float, b: float, n_points: int = 10000) -> float:
    """
    Tích phân Monte Carlo với QRNG
    """
    qrng_service = QuantumRandomService()
    sum_values = 0
    
    for _ in range(n_points):
        x = a + qrng_service.get_random_float() * (b - a)
        sum_values += func(x)
    
    integral = (b - a) * sum_values / n_points
    return integral

# Test quantum Monte Carlo
print("\nQuantum Monte Carlo Demo:")
print("Estimating π:")
pi_estimate = quantum_monte_carlo_pi(10000)
print(f"Estimated π: {pi_estimate:.6f}")
print(f"Actual π: {np.pi:.6f}")
print(f"Error: {abs(pi_estimate - np.pi):.6f}")

print("\nIntegrating x² from 0 to 1:")
def square_func(x):
    return x**2

integral = quantum_integration(square_func, 0, 1, 10000)
print(f"Estimated integral: {integral:.6f}")
print(f"Actual integral: 1/3 = {1/3:.6f}")
print(f"Error: {abs(integral - 1/3):.6f}")
```

## 📈 Performance Analysis

### 1. Speed Comparison

```python
def performance_comparison(n_bits: int = 10000):
    """
    So sánh hiệu suất các phương pháp QRNG
    """
    methods = {
        'Quantum Hadamard': lambda: qrng.generate_random_bits(n_bits, 'hadamard'),
        'Quantum Rotation': lambda: qrng.generate_random_bits(n_bits, 'rotation'),
        'Quantum Bell': lambda: qrng.generate_random_bits(n_bits, 'bell'),
        'Classical Python': lambda: [random.randint(0, 1) for _ in range(n_bits)],
        'NumPy': lambda: np.random.randint(0, 2, n_bits).tolist()
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        start_time = time.time()
        bits = method_func()
        end_time = time.time()
        
        # Test quality
        quality = tester.run_all_tests(bits)
        entropy = quality['entropy']['entropy']
        
        results[method_name] = {
            'time': end_time - start_time,
            'entropy': entropy,
            'bits_per_second': n_bits / (end_time - start_time)
        }
    
    return results

# Performance analysis
print("\nPerformance Analysis:")
perf_results = performance_comparison(1000)

for method, data in perf_results.items():
    print(f"\n{method}:")
    print(f"  Time: {data['time']:.4f} seconds")
    print(f"  Entropy: {data['entropy']:.4f}")
    print(f"  Bits/second: {data['bits_per_second']:.0f}")
```

## 🎯 Bài tập thực hành

### Bài tập 1: Implement Quantum Random Walk
```python
def quantum_random_walk():
    """
    Triển khai quantum random walk
    """
    # TODO: Implement quantum random walk
    pass
```

### Bài tập 2: Quantum Lottery System
```python
def quantum_lottery():
    """
    Hệ thống xổ số lượng tử
    """
    # TODO: Implement quantum lottery
    pass
```

### Bài tập 3: Quantum Art Generation
```python
def quantum_art_generation():
    """
    Tạo nghệ thuật sử dụng QRNG
    """
    # TODO: Implement quantum art generation
    pass
```

## 📚 Tài liệu tham khảo

1. **Quantum Random Number Generation**: Herrero-Collantes, M. & Garcia-Escartin, J.C. (2017). Quantum random number generators.
2. **Statistical Testing**: NIST Special Publication 800-22: A Statistical Test Suite for Random and Pseudorandom Number Generators.
3. **Cryptographic Applications**: Menezes, A.J., van Oorschot, P.C. & Vanstone, S.A. (1996). Handbook of Applied Cryptography.
4. **Monte Carlo Methods**: Metropolis, N. & Ulam, S. (1949). The Monte Carlo method.

## 🔮 Hướng dẫn tiếp theo

- **Day 33**: Quantum Simulation Projects
- **Day 34**: Quantum Computing on Real Hardware
- **Day 35**: Capstone Project và Portfolio Building

---

*"True randomness is a fundamental resource in quantum information processing."* - Artur Ekert 