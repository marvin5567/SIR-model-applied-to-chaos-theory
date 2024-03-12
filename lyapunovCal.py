import numpy as np


# sir model equations
def sir_model(t, y, params):
    beta_0, beta_1, L, k, t_0, gamma_0, alpha, N = params
    S, I, R = y
    
    dSdt = -beta_0 * (1 + beta_1 * np.cos(2 * np.pi * t)) * S * I / N * L / (1 + np.exp(-k * (t - t_0)))
    dIdt = beta_0 * (1 + beta_1 * np.cos(2 * np.pi * t)) * S * I / N - gamma_0 * (1 - alpha * I / N) * I
    dRdt = gamma_0 * (1 - alpha * I / N) * I
    
    return [dSdt, dIdt, dRdt]

def integrate_sir_model(params, initial_conditions, t_span, num_steps):
    t0, t1 = t_span
    t_values = np.linspace(t0, t1, num_steps + 1)
    delta_t = (t1 - t0) / num_steps
    
    S_values = np.zeros(num_steps + 1)
    I_values = np.zeros(num_steps + 1)
    R_values = np.zeros(num_steps + 1)
    
    S_values[0], I_values[0], R_values[0] = initial_conditions
    
    # runge kutta method is used
    for i in range(num_steps):
        t = t_values[i]
        y = [S_values[i], I_values[i], R_values[i]]
        
        k1 = np.array(sir_model(t, y, params))
        k2 = np.array(sir_model(t + delta_t / 2, y + delta_t / 2 * k1, params))
        k3 = np.array(sir_model(t + delta_t / 2, y + delta_t / 2 * k2, params))
        k4 = np.array(sir_model(t + delta_t, y + delta_t * k3, params))
        
        y_next = y + delta_t / 6 * (k1 + 2*k2 + 2*k3 + k4)
        S_values[i + 1], I_values[i + 1], R_values[i + 1] = y_next
    
    return t_values, S_values, I_values, R_values

# define parameters (params) and initial conditions
params = (0.8, 0.25, 1, 0.3, 9, 0.1, 0.25, 14000)
initial_conditions = (10000, 100, 0)  # S0, I0, R0
t_span = (0, 100) # basically start to finish time
num_steps = 1000

# perturb initial conditions
perturbation = 1e-6 # this value will also be used for d0 when calculating the probability horizon
initial_conditions_perturbed = [x + perturbation for x in initial_conditions]

# integrate the equations for original and perturbed initial conditions
t, S_orig, I_orig, R_orig = integrate_sir_model(params, initial_conditions, t_span, num_steps)
t, S_pert, I_pert, R_pert = integrate_sir_model(params, initial_conditions_perturbed, t_span, num_steps)

# calculate the separation distance between trajectories
separation_distance = np.sqrt((S_orig - S_pert) ** 2 + (I_orig - I_pert) ** 2 + (R_orig - R_pert) ** 2)

# fit exponential curve to separation distance versus time
fit_params = np.polyfit(t, np.log(separation_distance), 1)

# calculate Lyapunov exponent (slope of the fitted curve)
lyapunov_exponent = fit_params[0]

print("Lyapunov Exponent:", lyapunov_exponent)