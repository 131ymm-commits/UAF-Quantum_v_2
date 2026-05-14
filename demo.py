import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

hbar = 1.0  # для симуляций ставим = 1 (натуральные единицы)

def uaf_system(t, y, eta=1.0, lam=0.1, alpha0=0.1):
    Gamma, alpha = y
    # Простая двухуровневая модель
    dGamma_dt = (eta * (1 - alpha)**2 * Gamma - lam * alpha * Gamma) / hbar
    dalpha_dt = (0.3 * alpha * (1 - alpha) - 0.5 * (1 - alpha) * Gamma**2) / hbar
    return [dGamma_dt, dalpha_dt]

# Решение
sol = solve_ivp(uaf_system, [0, 50], [1.0, 0.05], t_eval=np.linspace(0, 50, 500))

plt.figure(figsize=(10,6))
plt.plot(sol.t, sol.y[0], label='Γ (когерентность)')
plt.plot(sol.t, sol.y[1], label='α (интеграция в прошлое)')
plt.xlabel('Внутреннее время τ')
plt.ylabel('Значение')
plt.legend()
plt.title('UAF-Q: Декогеренция и стрелка времени')
plt.grid()
plt.show()
