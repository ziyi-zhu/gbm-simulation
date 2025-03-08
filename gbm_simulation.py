import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Optional: for progress bars
import pandas as pd

def calculate_theoretical_probability(S0, mu, sigma, alpha, beta):
    """
    Calculate the theoretical probability of hitting upper boundary (alpha) before lower boundary (beta)
    under Geometric Brownian Motion.
    
    Parameters:
    S0 (float): Initial stock price
    mu (float): Drift parameter
    sigma (float): Volatility parameter
    alpha (float): Upper boundary (limit price)
    beta (float): Lower boundary (stop price)
    
    Returns:
    float: Probability of hitting alpha before beta
    """
    mu_tilde = mu/sigma - sigma/2
    Y_tau_alpha = (np.log(alpha) - np.log(S0))/sigma
    Y_tau_beta = (np.log(beta) - np.log(S0))/sigma
    
    numerator = 1 - np.exp(-2 * mu_tilde * Y_tau_beta)
    denominator = np.exp(-2 * mu_tilde * Y_tau_alpha) - np.exp(-2 * mu_tilde * Y_tau_beta)
    
    return numerator / denominator

def run_simulation(S0, mu, sigma, alpha, beta, num_simulations=10000, dt=0.001, max_time=100):
    """
    Run Monte Carlo simulation of Geometric Brownian Motion with absorbing boundaries.
    
    Parameters:
    S0 (float): Initial stock price
    mu (float): Drift parameter
    sigma (float): Volatility parameter
    alpha (float): Upper boundary (limit price)
    beta (float): Lower boundary (stop price)
    num_simulations (int): Number of simulation paths
    dt (float): Time step for simulation
    max_time (float): Maximum simulation time
    
    Returns:
    dict: Results including theoretical and empirical probabilities
    """
    # Calculate theoretical probability
    theoretical_prob = calculate_theoretical_probability(S0, mu, sigma, alpha, beta)
    
    # Monte Carlo simulation
    hit_alpha = 0
    hit_beta = 0
    no_hit = 0
    
    for _ in range(num_simulations):
        price = S0
        time = 0
        
        while time < max_time:
            # Generate random normal increment
            z = np.random.normal(0, 1)
            
            # Update price using GBM
            price = price * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
            
            # Check boundaries
            if price >= alpha:
                hit_alpha += 1
                break
            elif price <= beta:
                hit_beta += 1
                break
            
            time += dt
        
        # If simulation reached maxTime without hitting either boundary
        if time >= max_time:
            no_hit += 1
    
    total_complete = hit_alpha + hit_beta
    empirical_prob = hit_alpha / total_complete if total_complete > 0 else np.nan
    
    # Calculate error
    abs_error = abs(theoretical_prob - empirical_prob)
    rel_error = (abs_error / theoretical_prob * 100) if theoretical_prob != 0 else np.nan
    
    return {
        'theoretical': theoretical_prob,
        'empirical': empirical_prob,
        'abs_error': abs_error,
        'rel_error': rel_error,
        'hit_alpha': hit_alpha,
        'hit_beta': hit_beta,
        'no_hit': no_hit,
        'completion_rate': (total_complete / num_simulations) * 100
    }

def simulate_single_path(S0, mu, sigma, alpha, beta, dt=0.01, max_time=50):
    """
    Simulate a single price path and track if/when it hits a boundary.
    
    Returns:
    tuple: (path, outcome)
        path: list of (time, price) points
        outcome: 'hit-alpha', 'hit-beta', or 'no-hit'
    """
    path = [(0, S0)]
    price = S0
    time = 0
    outcome = None
    
    while time < max_time:
        time += dt
        # Generate random normal increment
        z = np.random.normal(0, 1)
        
        # Update price using GBM
        price = price * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        
        path.append((time, price))
        
        # Check boundaries
        if price >= alpha:
            outcome = 'hit-alpha'
            break
        elif price <= beta:
            outcome = 'hit-beta'
            break
    
    if not outcome:
        outcome = 'no-hit'
    
    return path, outcome

def run_comprehensive_tests():
    """
    Run comprehensive tests with various parameter combinations.
    """
    results = []
    
    # 1. Test with varying drift values
    print("\n1. VARYING DRIFT VALUES")
    print("S0 = 100, sigma = 0.2, alpha = 110, beta = 90")
    drifts = [-0.1, -0.05, 0, 0.05, 0.1]
    for drift in drifts:
        result = run_simulation(100, drift, 0.2, 110, 90)
        print(f"μ = {drift:.3f}: Theoretical = {result['theoretical']:.4f}, "
              f"Empirical = {result['empirical']:.4f}, "
              f"Error = {result['abs_error']:.4f} ({result['rel_error']:.2f}%)")
        results.append({
            'test_type': 'drift',
            'mu': drift,
            'sigma': 0.2,
            'S0': 100,
            'alpha': 110,
            'beta': 90,
            **result
        })
    
    # 2. Test with varying volatility values
    print("\n2. VARYING VOLATILITY VALUES")
    print("S0 = 100, mu = 0.05, alpha = 110, beta = 90")
    vols = [0.1, 0.2, 0.3, 0.4]
    for vol in vols:
        result = run_simulation(100, 0.05, vol, 110, 90)
        print(f"σ = {vol:.2f}: Theoretical = {result['theoretical']:.4f}, "
              f"Empirical = {result['empirical']:.4f}, "
              f"Error = {result['abs_error']:.4f} ({result['rel_error']:.2f}%)")
        results.append({
            'test_type': 'volatility',
            'mu': 0.05,
            'sigma': vol,
            'S0': 100,
            'alpha': 110,
            'beta': 90,
            **result
        })
    
    # 3. Test with varying boundary distances
    print("\n3. VARYING BOUNDARY CONFIGURATIONS")
    print("S0 = 100, mu = 0.05, sigma = 0.2")
    configs = [
        {'alpha': 105, 'beta': 95},
        {'alpha': 110, 'beta': 90},
        {'alpha': 120, 'beta': 80},
        {'alpha': 120, 'beta': 90},
        {'alpha': 110, 'beta': 80}
    ]
    for config in configs:
        result = run_simulation(100, 0.05, 0.2, config['alpha'], config['beta'])
        print(f"α = {config['alpha']}, β = {config['beta']}: Theoretical = {result['theoretical']:.4f}, "
              f"Empirical = {result['empirical']:.4f}, "
              f"Error = {result['abs_error']:.4f} ({result['rel_error']:.2f}%)")
        results.append({
            'test_type': 'boundaries',
            'mu': 0.05,
            'sigma': 0.2,
            'S0': 100,
            'alpha': config['alpha'],
            'beta': config['beta'],
            **result
        })
    
    # Create a DataFrame with all results
    return pd.DataFrame(results)

def plot_sample_paths(S0=100, mu=0.05, sigma=0.2, alpha=110, beta=90, n_paths=5):
    """
    Generate and plot sample paths of the GBM process with absorbing boundaries.
    """
    plt.figure(figsize=(12, 6))
    
    for i in range(n_paths):
        path, outcome = simulate_single_path(S0, mu, sigma, alpha, beta)
        times, prices = zip(*path)
        
        if outcome == 'hit-alpha':
            label = f"Path {i+1}: Hit limit (α)"
            linestyle = '-'
        elif outcome == 'hit-beta':
            label = f"Path {i+1}: Hit stop (β)"
            linestyle = '--'
        else:
            label = f"Path {i+1}: No hit"
            linestyle = ':'
        
        plt.plot(times, prices, label=label, linestyle=linestyle)
    
    # Add horizontal lines for boundaries
    plt.axhline(y=alpha, color='g', linestyle='-', alpha=0.5, label='Limit (α)')
    plt.axhline(y=beta, color='r', linestyle='-', alpha=0.5, label='Stop (β)')
    plt.axhline(y=S0, color='k', linestyle=':', alpha=0.3, label='Initial Price')
    
    plt.title(f'Sample GBM Paths (μ={mu}, σ={sigma})')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt

def plot_parameter_impact():
    """
    Create plots showing the impact of different parameters on the probability.
    """
    # 1. Impact of drift-to-volatility ratio
    mu_sigma_ratios = np.linspace(-2, 2, 41)
    probs = []
    
    for ratio in mu_sigma_ratios:
        mu = ratio * 0.2  # Fix sigma at 0.2
        prob = calculate_theoretical_probability(100, mu, 0.2, 110, 90)
        probs.append(prob)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_sigma_ratios, probs, 'b-', linewidth=2)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.xlabel('μ/σ Ratio')
    plt.ylabel('Probability of Hitting Upper Boundary First')
    plt.title('Impact of Drift-to-Volatility Ratio on Boundary Hitting Probability')
    
    # 2. Contour plot of probabilities over different alpha and beta values
    alpha_values = np.linspace(101, 120, 20)
    beta_values = np.linspace(80, 99, 20)
    alpha_grid, beta_grid = np.meshgrid(alpha_values, beta_values)
    prob_grid = np.zeros_like(alpha_grid)
    
    for i in range(len(beta_values)):
        for j in range(len(alpha_values)):
            alpha = alpha_grid[i, j]
            beta = beta_grid[i, j]
            prob_grid[i, j] = calculate_theoretical_probability(100, 0.05, 0.2, alpha, beta)
    
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(alpha_grid, beta_grid, prob_grid, 50, cmap='viridis')
    plt.colorbar(contour, label='Probability of Hitting Upper Boundary')
    plt.xlabel('Upper Boundary (α)')
    plt.ylabel('Lower Boundary (β)')
    plt.title('Probability Contour Map for Different Boundary Combinations')
    plt.grid(True, alpha=0.3)
    
    return plt

if __name__ == "__main__":
    # Run basic simulation
    S0 = 100
    mu = 0.05
    sigma = 0.2
    alpha = 110
    beta = 90
    
    # Calculate theoretical probability
    theoretical_prob = calculate_theoretical_probability(S0, mu, sigma, alpha, beta)
    print(f"Theoretical probability of hitting α={alpha} before β={beta}: {theoretical_prob:.4f}")
    
    # Run simulation
    result = run_simulation(S0, mu, sigma, alpha, beta, num_simulations=20000)
    print(f"Empirical probability from simulation: {result['empirical']:.4f}")
    print(f"Absolute error: {result['abs_error']:.4f}")
    print(f"Relative error: {result['rel_error']:.2f}%")
    print(f"Completion rate: {result['completion_rate']:.2f}%")
    
    # Run comprehensive tests
    results_df = run_comprehensive_tests()
    
    # Plot sample paths and parameter impacts
    plot_sample_paths()
    plot_parameter_impact()
    plt.show()
