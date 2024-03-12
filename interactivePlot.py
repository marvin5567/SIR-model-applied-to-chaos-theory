import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import ipywidgets as widgets
from ipywidgets import interactive, HBox, VBox

# entire sir model
def sir_model(t, y, beta_0, beta_1, gamma_0, alpha, N, L, k, t_0):
    S, I, R = y # each value will be plotted against time
    dSdt = -beta_0 * (1 + beta_1 * np.cos(2 * np.pi * t)) * S * I / N * L / (1 + np.exp(-k * (t - t_0))) # susceptible
    dIdt = beta_0 * (1 + beta_1 * np.cos(2 * np.pi * t)) * S * I / N - gamma_0 * (1 - alpha * I / N) * I # infected
    dRdt = gamma_0 * (1 - alpha * I / N) * I # recovered
    return [dSdt, dIdt, dRdt]

# function to compute the derivatives for the vector field
def compute_derivatives(S, I, beta_0, beta_1, gamma_0, alpha, N, L, k, t_0, t): # vector field will be susceptible vs infected
    dSdt = -beta_0 * (1 + beta_1 * np.cos(2 * np.pi * t)) * S * I / N * L / (1 + np.exp(-k * (t - t_0)))
    dIdt = beta_0 * (1 + beta_1 * np.cos(2 * np.pi * t)) * S * I / N - gamma_0 * (1 - alpha * I / N) * I
    return dSdt, dIdt

# plotting function
def plot_sir_and_vector_field(beta_0, beta_1, gamma_0, alpha, N, L, k, t_0, S0, I0, R0, t_end):
    # solve the SIR model
    y0 = [S0, I0, R0]
    t_span = [0, t_end]
    sol = solve_ivp(sir_model, t_span, y0, args=(beta_0, beta_1, gamma_0, alpha, N, L, k, t_0), t_eval=np.linspace(0, t_end, 1000))

    # plotting the model
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    
    ax[0].plot(sol.t, sol.y[0], label='Susceptible')
    ax[0].plot(sol.t, sol.y[1], label='Infected')
    ax[0].plot(sol.t, sol.y[2], label='Recovered')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Population')
    ax[0].set_title('SIR Model')
    ax[0].legend()
    ax[0].grid(True)

    # creating vector field
    S_range = np.linspace(0, N, 20)
    I_range = np.linspace(0, N, 20)
    S_grid, I_grid = np.meshgrid(S_range, I_range)
    dSdt_grid, dIdt_grid = compute_derivatives(S_grid, I_grid, beta_0, beta_1, gamma_0, alpha, N, L, k, t_0, 0)

    # plotting the vector field
    ax[1].quiver(S_grid, I_grid, dSdt_grid, dIdt_grid)
    ax[1].set_xlabel('Susceptible Population (S)')
    ax[1].set_ylabel('Infected Population (I)')
    ax[1].set_title('Vector Field for S vs I')
    ax[1].axis([0, N, 0, N])
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()

# since i find it annoying to change each value, im gonna setup sliders for this
beta_0_slider = widgets.FloatSlider(value=0.5, min=0, max=1, step=0.01, description='Beta 0:')
beta_1_slider = widgets.FloatSlider(value=0.2, min=0, max=1, step=0.01, description='Beta 1:')
gamma_0_slider = widgets.FloatSlider(value=0.1, min=0, max=1, step=0.01, description='Gamma 0:')
alpha_slider = widgets.FloatSlider(value=0.1, min=0, max=1, step=0.01, description='Alpha:')
N_slider = widgets.FloatSlider(value=16000, min=1, max=100000, step=100, description='Total Population (N):')
L_slider = widgets.FloatSlider(value=0.9, min=0, max=1, step=0.01, description='L:')
k_slider = widgets.FloatSlider(value=0.2, min=0, max=1, step=0.01, description='k:')
t_0_slider = widgets.FloatSlider(value=5, min=0, max=10, step=0.1, description='t_0:')
S0_slider = widgets.FloatSlider(value=9900, min=0, max=10000, step=100, description='S0:')
I0_slider = widgets.FloatSlider(value=10, min=0, max=1000, step=10, description='I0:')
R0_slider = widgets.FloatSlider(value=0, min=0, max=1000, step=10, description='R0:')
t_end_slider = widgets.FloatSlider(value=100, min=0, max=1000, step=10, description='End Time:')

# setup interactive plot
interactive_plot = interactive(plot_sir_and_vector_field, beta_0=beta_0_slider, beta_1=beta_1_slider, gamma_0=gamma_0_slider,
                               alpha=alpha_slider, N=N_slider, L=L_slider, k=k_slider, t_0=t_0_slider,
                               S0=S0_slider, I0=I0_slider, R0=R0_slider, t_end=t_end_slider)

display(interactive_plot) # displays the plot; notebook only command :(
