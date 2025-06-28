# Day 26: Shor's Algorithm for Factoring

## 🎯 Mục tiêu
- Hiểu nguyên lý hoạt động của Shor's Algorithm
- Triển khai quantum period finding
- Áp dụng modular arithmetic trong quantum computing
- Hiểu tác động đến RSA cryptography
- Giải quyết các thách thức implementation

## 🧠 Shor's Algorithm - Tổng Quan

### Tại sao Shor's Algorithm quan trọng?
- **Cryptographic breakthrough**: Có thể phá vỡ RSA encryption
- **Quantum advantage**: Giải quyết factoring nhanh hơn classical algorithms
- **Period finding**: Ứng dụng quantum Fourier transform
- **Modular arithmetic**: Kết hợp classical và quantum computation

```python
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.algorithms import Shor
from qiskit.algorithms.factorizers import Shor as ShorFactorizer
from qiskit.circuit.library import QFT
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
import math
```

## 🔧 Mathematical Foundation

### 1. Factoring Problem và RSA

```python
def rsa_factoring_problem():
    """
    Giải thích bài toán factoring và RSA
    """
    # RSA: n = p * q (public key)
    # Private key: d = e^(-1) mod φ(n)
    # φ(n) = (p-1)(q-1)
    
    # Example RSA parameters
    p = 61
    q = 53
    n = p * q
    phi_n = (p - 1) * (q - 1)
    e = 17  # Public exponent
    
    print(f"RSA Parameters:")
    print(f"p = {p}, q = {q}")
    print(f"n = p * q = {n}")
    print(f"φ(n) = (p-1)(q-1) = {phi_n}")
    print(f"Public key: (n={n}, e={e})")
    
    # If we can factor n, we can find p and q
    # Then calculate φ(n) and find private key d
    d = pow(e, -1, phi_n)
    print(f"Private key d = {d}")
    
    return n, e, d, p, q

# Demo RSA
n, e, d, p, q = rsa_factoring_problem()
```

### 2. Period Finding Problem

```python
def period_finding_demo():
    """
    Demo period finding - core của Shor's algorithm
    """
    # Given: f(x) = a^x mod N
    # Find: smallest r such that f(x+r) = f(x)
    
    N = 15  # Number to factor
    a = 7   # Base for modular exponentiation
    
    print(f"Period Finding for f(x) = {a}^x mod {N}")
    
    # Calculate function values
    function_values = []
    for x in range(20):
        value = pow(a, x, N)
        function_values.append(value)
        print(f"f({x}) = {a}^{x} mod {N} = {value}")
    
    # Find period
    period = None
    for r in range(1, len(function_values)):
        if function_values[r] == function_values[0]:
            period = r
            break
    
    print(f"\nPeriod r = {period}")
    
    # Verify period
    if period:
        print(f"Verification: f(0) = {function_values[0]}, f({period}) = {function_values[period]}")
    
    return function_values, period

# Demo period finding
func_values, period = period_finding_demo()
```

### 3. Quantum Period Finding

```python
def quantum_period_finding_circuit(N, a, precision_qubits=8):
    """
    Tạo quantum circuit cho period finding
    """
    # Number of qubits needed
    n = len(bin(N)[2:])  # Bits needed to represent N
    total_qubits = precision_qubits + n
    
    # Create circuit
    qc = QuantumCircuit(total_qubits, precision_qubits)
    
    # Initialize counting register to |0⟩
    # Initialize work register to |1⟩
    qc.x(precision_qubits)
    
    # Apply Hadamard to counting register
    for i in range(precision_qubits):
        qc.h(i)
    
    # Apply controlled modular exponentiation
    # This is the key part of Shor's algorithm
    for i in range(precision_qubits):
        # Apply controlled-U^(2^i) where U|x⟩ = |ax mod N⟩
        # This is simplified - actual implementation is more complex
        qc.cx(i, precision_qubits)
    
    # Apply inverse QFT to counting register
    qft = QFT(precision_qubits)
    qc.compose(qft.inverse(), qubits=range(precision_qubits), inplace=True)
    
    # Measure counting register
    qc.measure(range(precision_qubits), range(precision_qubits))
    
    print("Quantum Period Finding Circuit:")
    print(qc)
    
    return qc

# Tạo quantum period finding circuit
period_circuit = quantum_period_finding_circuit(15, 7)
```

## 🎯 Shor's Algorithm Implementation

### 1. Classical Part - GCD và Modular Arithmetic

```python
def classical_shor_components():
    """
    Các thành phần classical của Shor's algorithm
    """
    def gcd(a, b):
        """Euclidean algorithm for GCD"""
        while b:
            a, b = b, a % b
        return a
    
    def mod_pow(base, exponent, modulus):
        """Modular exponentiation using square-and-multiply"""
        result = 1
        base = base % modulus
        
        while exponent > 0:
            if exponent % 2 == 1:
                result = (result * base) % modulus
            base = (base * base) % modulus
            exponent //= 2
        
        return result
    
    def find_period_classical(a, N):
        """Find period of f(x) = a^x mod N using classical method"""
        seen = {}
        x = 1
        
        for i in range(N):
            if x in seen:
                return i - seen[x]
            seen[x] = i
            x = (x * a) % N
        
        return None
    
    # Test classical components
    N = 15
    a = 7
    
    print(f"Testing classical components for N={N}, a={a}")
    print(f"GCD({a}, {N}) = {gcd(a, N)}")
    print(f"{a}^5 mod {N} = {mod_pow(a, 5, N)}")
    
    period = find_period_classical(a, N)
    print(f"Period of f(x) = {a}^x mod {N}: {period}")
    
    return gcd, mod_pow, find_period_classical

# Test classical components
gcd_func, mod_pow_func, find_period_func = classical_shor_components()
```

### 2. Quantum Part - Phase Estimation

```python
def quantum_phase_estimation_demo():
    """
    Demo quantum phase estimation cho period finding
    """
    def create_phase_estimation_circuit(phase, precision_qubits=4):
        """Create phase estimation circuit for given phase"""
        qc = QuantumCircuit(precision_qubits + 1, precision_qubits)
        
        # Initialize eigenstate |u⟩
        qc.x(precision_qubits)
        
        # Apply Hadamard to counting qubits
        for i in range(precision_qubits):
            qc.h(i)
        
        # Apply controlled phase rotations
        for i in range(precision_qubits):
            # Apply U^(2^i) where U|u⟩ = e^(2πiφ)|u⟩
            angle = 2 * np.pi * phase * (2**i)
            qc.cp(angle, i, precision_qubits)
        
        # Apply inverse QFT
        qft = QFT(precision_qubits)
        qc.compose(qft.inverse(), qubits=range(precision_qubits), inplace=True)
        
        # Measure
        qc.measure(range(precision_qubits), range(precision_qubits))
        
        return qc
    
    # Test with known phase
    phase = 0.125  # 1/8
    qc = create_phase_estimation_circuit(phase)
    
    print(f"Phase Estimation Circuit for φ = {phase}")
    print(qc)
    
    # Execute
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=1000)
    result = job.result()
    counts = result.get_counts()
    
    print(f"Measurement results: {counts}")
    
    # Convert measurement to phase estimate
    for bitstring, count in counts.items():
        if count > 50:  # Significant result
            measured_phase = int(bitstring, 2) / (2**4)
            print(f"Measured phase: {measured_phase} (expected: {phase})")
    
    return qc, counts

# Demo phase estimation
phase_circuit, phase_counts = quantum_phase_estimation_demo()
```

### 3. Complete Shor's Algorithm

```python
def shor_algorithm_implementation(N, a=None):
    """
    Triển khai Shor's algorithm hoàn chỉnh
    """
    if a is None:
        # Choose random a
        a = np.random.randint(2, N)
    
    print(f"Shor's Algorithm for N = {N}, a = {a}")
    
    # Step 1: Check if N is even
    if N % 2 == 0:
        return 2, N // 2
    
    # Step 2: Check if N is a perfect power
    for k in range(2, int(np.log2(N)) + 1):
        root = int(N ** (1/k))
        if root ** k == N:
            return root, N // root
    
    # Step 3: Choose random a and check GCD
    while True:
        a = np.random.randint(2, N)
        gcd_val = math.gcd(a, N)
        if gcd_val > 1:
            return gcd_val, N // gcd_val
        if gcd_val == 1:
            break
    
    # Step 4: Find period using quantum period finding
    # This is the quantum part - simplified for demonstration
    print(f"Finding period of f(x) = {a}^x mod {N}")
    
    # Classical period finding (in practice, this would be quantum)
    period = find_period_classical(a, N)
    
    if period is None or period % 2 == 1:
        print("Period not found or odd - try different a")
        return None, None
    
    # Step 5: Use period to find factors
    half_period = period // 2
    candidate = pow(a, half_period, N)
    
    if candidate == N - 1:
        print("Period gives no useful information - try different a")
        return None, None
    
    factor1 = math.gcd(candidate + 1, N)
    factor2 = math.gcd(candidate - 1, N)
    
    if factor1 > 1 and factor1 < N:
        return factor1, N // factor1
    elif factor2 > 1 and factor2 < N:
        return factor2, N // factor2
    else:
        print("No factors found - try different a")
        return None, None

# Test Shor's algorithm
test_numbers = [15, 21, 33]
for N in test_numbers:
    print(f"\n{'='*50}")
    factor1, factor2 = shor_algorithm_implementation(N)
    if factor1:
        print(f"Factors of {N}: {factor1} and {factor2}")
        print(f"Verification: {factor1} * {factor2} = {factor1 * factor2}")
    else:
        print(f"Could not factor {N}")
```

## 🔐 RSA Cryptography và Quantum Threat

### 1. RSA Encryption/Decryption

```python
def rsa_cryptography_demo():
    """
    Demo RSA encryption và quantum threat
    """
    # Generate RSA keys
    p = 61
    q = 53
    n = p * q
    phi_n = (p - 1) * (q - 1)
    e = 17
    d = pow(e, -1, phi_n)
    
    print(f"RSA Key Generation:")
    print(f"Public key: (n={n}, e={e})")
    print(f"Private key: d={d}")
    
    # Encrypt message
    message = 123
    encrypted = pow(message, e, n)
    decrypted = pow(encrypted, d, n)
    
    print(f"\nEncryption/Decryption:")
    print(f"Original message: {message}")
    print(f"Encrypted: {encrypted}")
    print(f"Decrypted: {decrypted}")
    
    # Quantum threat simulation
    print(f"\nQuantum Threat Analysis:")
    print(f"To break RSA, we need to factor n = {n}")
    print(f"Classical factoring complexity: O(exp(n^(1/3)))")
    print(f"Shor's algorithm complexity: O((log n)^3)")
    
    # Simulate quantum factoring
    factor1, factor2 = shor_algorithm_implementation(n)
    if factor1:
        print(f"Quantum factoring found factors: {factor1}, {factor2}")
        print(f"Private key can be calculated: d = e^(-1) mod φ(n)")
        print(f"where φ(n) = (p-1)(q-1) = ({factor1}-1)({factor2}-1)")
    
    return n, e, d, encrypted, decrypted

# Demo RSA và quantum threat
rsa_n, rsa_e, rsa_d, rsa_enc, rsa_dec = rsa_cryptography_demo()
```

### 2. Post-Quantum Cryptography

```python
def post_quantum_cryptography():
    """
    Giới thiệu post-quantum cryptography
    """
    print("Post-Quantum Cryptography Alternatives:")
    
    alternatives = {
        "Lattice-based": {
            "examples": ["NTRU", "LWE", "SIS"],
            "security": "Based on lattice problems",
            "quantum_resistant": "Yes"
        },
        "Code-based": {
            "examples": ["McEliece", "Niederreiter"],
            "security": "Based on error-correcting codes",
            "quantum_resistant": "Yes"
        },
        "Hash-based": {
            "examples": ["XMSS", "SPHINCS+"],
            "security": "Based on hash functions",
            "quantum_resistant": "Yes"
        },
        "Multivariate": {
            "examples": ["Rainbow", "HFE"],
            "security": "Based on multivariate polynomials",
            "quantum_resistant": "Yes"
        }
    }
    
    for method, details in alternatives.items():
        print(f"\n{method}:")
        print(f"  Examples: {', '.join(details['examples'])}")
        print(f"  Security: {details['security']}")
        print(f"  Quantum Resistant: {details['quantum_resistant']}")
    
    return alternatives

# Demo post-quantum cryptography
pqc_methods = post_quantum_cryptography()
```

## 🔄 Implementation Challenges

### 1. Quantum Error Correction

```python
def quantum_error_correction_shor():
    """
    Error correction challenges cho Shor's algorithm
    """
    print("Quantum Error Correction Challenges for Shor's Algorithm:")
    
    challenges = {
        "Coherence Time": {
            "issue": "Qubits lose coherence during long computations",
            "impact": "Limits circuit depth and accuracy",
            "solution": "Error correction codes, surface codes"
        },
        "Gate Errors": {
            "issue": "Imperfect quantum gates introduce errors",
            "impact": "Accumulates during computation",
            "solution": "Fault-tolerant quantum computing"
        },
        "Measurement Errors": {
            "issue": "Imperfect measurements affect results",
            "impact": "Reduces success probability",
            "solution": "Repeated measurements, error mitigation"
        },
        "Scalability": {
            "issue": "Large number of qubits needed for factoring",
            "impact": "Current devices insufficient",
            "solution": "Modular quantum computing, distributed algorithms"
        }
    }
    
    for challenge, details in challenges.items():
        print(f"\n{challenge}:")
        print(f"  Issue: {details['issue']}")
        print(f"  Impact: {details['impact']}")
        print(f"  Solution: {details['solution']}")
    
    return challenges

# Demo error correction challenges
error_challenges = quantum_error_correction_shor()
```

### 2. Resource Requirements

```python
def resource_requirements_analysis():
    """
    Phân tích resource requirements cho Shor's algorithm
    """
    print("Resource Requirements for Shor's Algorithm:")
    
    # Estimate qubits needed for different key sizes
    key_sizes = [512, 1024, 2048, 4096]
    
    for key_size in key_sizes:
        # Rough estimate: 2n qubits for n-bit number
        qubits_needed = 2 * key_size
        # Additional qubits for error correction
        error_correction_qubits = qubits_needed * 100  # Conservative estimate
        
        print(f"\nRSA-{key_size}:")
        print(f"  Logical qubits: {qubits_needed}")
        print(f"  Physical qubits (with error correction): {error_correction_qubits}")
        print(f"  Circuit depth: ~{key_size**3} gates")
        print(f"  Estimated runtime: ~{key_size**3} seconds (ideal)")
    
    # Current quantum computer capabilities
    print(f"\nCurrent Quantum Computer Capabilities:")
    print(f"  IBM Q System: ~100 qubits")
    print(f"  Google Sycamore: ~53 qubits")
    print(f"  Error rates: ~0.1% per gate")
    print(f"  Coherence time: ~100 microseconds")
    
    return key_sizes

# Demo resource requirements
resource_analysis = resource_requirements_analysis()
```

## 📊 Performance Analysis

### 1. Classical vs Quantum Complexity

```python
def complexity_comparison():
    """
    So sánh complexity classical vs quantum factoring
    """
    import time
    
    def trial_division(n):
        """Classical trial division"""
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors
    
    def quantum_factoring_simulation(n):
        """Simulate quantum factoring (simplified)"""
        # This is a simplified simulation
        # Real quantum factoring would use actual quantum hardware
        time.sleep(0.1)  # Simulate quantum computation time
        return shor_algorithm_implementation(n)
    
    # Test numbers
    test_numbers = [15, 21, 33, 35, 39]
    
    print("Classical vs Quantum Factoring Performance:")
    print(f"{'Number':<8} {'Classical (ms)':<15} {'Quantum (ms)':<15} {'Speedup':<10}")
    print("-" * 50)
    
    for n in test_numbers:
        # Classical timing
        start_time = time.time()
        classical_factors = trial_division(n)
        classical_time = (time.time() - start_time) * 1000
        
        # Quantum timing (simulated)
        start_time = time.time()
        quantum_factors = quantum_factoring_simulation(n)
        quantum_time = (time.time() - start_time) * 1000
        
        speedup = classical_time / quantum_time if quantum_time > 0 else float('inf')
        
        print(f"{n:<8} {classical_time:<15.2f} {quantum_time:<15.2f} {speedup:<10.2f}x")
    
    return test_numbers

# Demo complexity comparison
complexity_results = complexity_comparison()
```

### 2. Success Probability Analysis

```python
def success_probability_analysis():
    """
    Phân tích success probability của Shor's algorithm
    """
    def analyze_success_probability(N, num_trials=100):
        """Analyze success probability for given N"""
        successes = 0
        
        for _ in range(num_trials):
            factor1, factor2 = shor_algorithm_implementation(N)
            if factor1 is not None:
                successes += 1
        
        success_rate = successes / num_trials
        return success_rate
    
    # Test different numbers
    test_numbers = [15, 21, 33, 35, 39, 49, 55, 65]
    
    print("Success Probability Analysis:")
    print(f"{'Number':<8} {'Success Rate':<15} {'Factors':<15}")
    print("-" * 40)
    
    for N in test_numbers:
        success_rate = analyze_success_probability(N, num_trials=50)
        factor1, factor2 = shor_algorithm_implementation(N)
        factors_str = f"{factor1}, {factor2}" if factor1 else "None"
        
        print(f"{N:<8} {success_rate:<15.3f} {factors_str:<15}")
    
    return test_numbers

# Demo success probability analysis
success_analysis = success_probability_analysis()
```

## 📚 Bài Tập Thực Hành

### Bài tập 1: Implement Modular Exponentiation
```python
def modular_exponentiation_exercise():
    """
    Bài tập implement modular exponentiation
    """
    def mod_exp_naive(base, exponent, modulus):
        """Naive implementation"""
        result = 1
        for _ in range(exponent):
            result = (result * base) % modulus
        return result
    
    def mod_exp_efficient(base, exponent, modulus):
        """Efficient square-and-multiply implementation"""
        result = 1
        base = base % modulus
        
        while exponent > 0:
            if exponent % 2 == 1:
                result = (result * base) % modulus
            base = (base * base) % modulus
            exponent //= 2
        
        return result
    
    # Test cases
    test_cases = [
        (2, 10, 1000),
        (3, 7, 17),
        (5, 13, 23),
        (7, 8, 15)
    ]
    
    print("Modular Exponentiation Exercise:")
    print(f"{'Base':<6} {'Exp':<6} {'Mod':<6} {'Naive':<10} {'Efficient':<10} {'Match':<6}")
    print("-" * 50)
    
    for base, exp, mod in test_cases:
        naive_result = mod_exp_naive(base, exp, mod)
        efficient_result = mod_exp_efficient(base, exp, mod)
        match = "✓" if naive_result == efficient_result else "✗"
        
        print(f"{base:<6} {exp:<6} {mod:<6} {naive_result:<10} {efficient_result:<10} {match:<6}")
    
    return test_cases

# Chạy modular exponentiation exercise
mod_exp_exercise = modular_exponentiation_exercise()
```

### Bài tập 2: Period Finding Implementation
```python
def period_finding_exercise():
    """
    Bài tập implement period finding
    """
    def find_period_brute_force(a, N):
        """Brute force period finding"""
        seen = {}
        x = 1
        
        for i in range(N):
            if x in seen:
                return i - seen[x]
            seen[x] = i
            x = (x * a) % N
        
        return None
    
    def find_period_quantum_simulation(a, N, precision=8):
        """Simulate quantum period finding"""
        # This is a simplified simulation
        # Real quantum period finding would use QFT
        
        # Create superposition of states
        states = []
        for x in range(2**precision):
            value = pow(a, x, N)
            states.append((x, value))
        
        # Find period using frequency analysis
        value_counts = {}
        for x, value in states:
            if value in value_counts:
                value_counts[value].append(x)
            else:
                value_counts[value] = [x]
        
        # Find most common period
        periods = []
        for value, x_list in value_counts.items():
            if len(x_list) > 1:
                for i in range(1, len(x_list)):
                    period = x_list[i] - x_list[i-1]
                    periods.append(period)
        
        if periods:
            return min(periods)
        return None
    
    # Test cases
    test_cases = [
        (7, 15),
        (2, 15),
        (3, 10),
        (5, 12)
    ]
    
    print("Period Finding Exercise:")
    print(f"{'a':<4} {'N':<4} {'Brute Force':<12} {'Quantum Sim':<12}")
    print("-" * 35)
    
    for a, N in test_cases:
        brute_period = find_period_brute_force(a, N)
        quantum_period = find_period_quantum_simulation(a, N)
        
        print(f"{a:<4} {N:<4} {brute_period:<12} {quantum_period:<12}")
    
    return test_cases

# Chạy period finding exercise
period_exercise = period_finding_exercise()
```

### Bài tập 3: RSA Breaking Simulation
```python
def rsa_breaking_simulation():
    """
    Bài tập simulate breaking RSA với Shor's algorithm
    """
    def generate_rsa_keys(bit_length=8):
        """Generate small RSA keys for demonstration"""
        # Generate small primes
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        
        # Choose two random primes
        p = np.random.choice(primes)
        q = np.random.choice(primes)
        while q == p:
            q = np.random.choice(primes)
        
        n = p * q
        phi_n = (p - 1) * (q - 1)
        e = 3  # Small exponent for simplicity
        
        # Calculate private key
        d = pow(e, -1, phi_n)
        
        return n, e, d, p, q
    
    def break_rsa_with_shor(n, e):
        """Attempt to break RSA using Shor's algorithm"""
        print(f"Attempting to break RSA with n = {n}, e = {e}")
        
        # Try Shor's algorithm
        factor1, factor2 = shor_algorithm_implementation(n)
        
        if factor1:
            print(f"Success! Found factors: {factor1}, {factor2}")
            
            # Calculate private key
            phi_n = (factor1 - 1) * (factor2 - 1)
            d = pow(e, -1, phi_n)
            
            print(f"Private key d = {d}")
            return d
        else:
            print("Failed to find factors")
            return None
    
    # Generate RSA keys
    n, e, d_original, p, q = generate_rsa_keys()
    
    print(f"Generated RSA Keys:")
    print(f"Public key: (n={n}, e={e})")
    print(f"Private key: d={d_original}")
    print(f"Actual factors: p={p}, q={q}")
    
    # Attempt to break
    d_recovered = break_rsa_with_shor(n, e)
    
    if d_recovered:
        print(f"Recovery successful: {d_recovered == d_original}")
    
    return n, e, d_original, d_recovered

# Chạy RSA breaking simulation
rsa_breaking = rsa_breaking_simulation()
```

## 🎯 Kết Quả Mong Đợi
- Hiểu rõ nguyên lý Shor's algorithm và ứng dụng cho factoring
- Có thể triển khai quantum period finding
- Hiểu tác động đến RSA cryptography
- Giải quyết được các thách thức implementation

## 📖 Tài Liệu Tham Khảo
- [Shor's Algorithm Paper](https://arxiv.org/abs/quant-ph/9508027)
- [Qiskit Shor Implementation](https://qiskit.org/documentation/stubs/qiskit.algorithms.factorizers.Shor.html)
- [RSA Cryptography](https://en.wikipedia.org/wiki/RSA_(cryptosystem))
- [Post-Quantum Cryptography](https://csrc.nist.gov/projects/post-quantum-cryptography) 