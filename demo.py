import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =============================================
# UAF-Quantum: Двухуровневая система (Qubit)
# =============================================

hbar = 1.0  # Натуральные единицы

def uaf_two_level(t, y, Omega=0.5, Delta=0.0, lam=0.05, eta=1.0):
    """
    y = [Gamma, alpha, phi]
    Gamma - когерентность (аналог |<0|1>|)
    alpha - степень интеграции в прошлое (декогеренция)
    phi   - относительная фаза
    """
    Gamma, alpha, phi = y
    
    # Эффективная Rabi-частота
    Omega_eff = Omega * (1 - alpha)  # декогеренция ослабляет когерентное управление
    
    # Уравнения UAF-Q
    dGamma_dt = (eta * (1 - alpha)**2 * Gamma * np.cos(phi)          # когерентный рост
                 - lam * alpha * Gamma) / hbar                       # декогеренция
    
    dalpha_dt = (0.08 * alpha * (1 - alpha)                          # спонтанная интеграция
                 + 0.6 * (1 - alpha) * (1 - Gamma**2)) / hbar       # потеря когерентности → рост α
    
    # Эволюция фазы (детuning + Rabi)
    dphi_dt = Delta + Omega_eff * (1 - Gamma) / (Gamma + 1e-8)
    
    return [dGamma_dt, dalpha_dt, dphi_dt]


# Начальные условия
y0 = [0.95, 0.02, 0.0]   # высокая начальная когерентность

# Решение
t_span = (0, 80)
t_eval = np.linspace(0, 80, 1200)

sol = solve_ivp(uaf_two_level, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-6)

# ====================== Визуализация ======================
Gamma = sol.y[0]
alpha = sol.y[1]
phi = sol.y[2]

# Населённости (примерное)
P_excited = 0.5 * (1 - Gamma * np.cos(phi))   # грубая аппроксимация

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(sol.t, Gamma, 'b-', linewidth=2, label=r'$\Gamma$ — Когерентность (t_self)')
plt.plot(sol.t, alpha, 'r-', linewidth=2, label=r'$\alpha$ — Интеграция в прошлое')
plt.xlabel('Внутреннее время $\\tau$')
plt.ylabel('Значение')
plt.title('UAF-Q: Когерентность и Декогеренция')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(sol.t, P_excited, 'g-', linewidth=2, label='Населённость возбуждённого состояния')
plt.xlabel('Внутреннее время $\\tau$')
plt.ylabel('Вероятность')
plt.title('Динамика населённостей')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(sol.t, np.cos(phi) * Gamma, 'purple', label='Re(когерентность)')
plt.plot(sol.t, np.sin(phi) * Gamma, 'orange', label='Im(когерентность)')
plt.xlabel('Внутреннее время $\\tau$')
plt.ylabel('Когерентная амплитуда')
plt.title('Осцилляции когерентности')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(Gamma, alpha, 'k-', label='Траектория в фазовом пространстве')
plt.xlabel(r'$\Gamma$')
plt.ylabel(r'$\alpha$')
plt.title('Фазовый портрет')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Итоговая информация
print("=== Результат симуляции ===")
print(f"Финальная когерентность (Γ): {Gamma[-1]:.4f}")
print(f"Финальная интеграция (α):   {alpha[-1]:.4f}")
print(f"Средняя когерентность:      {np.mean(Gamma):.4f}")
