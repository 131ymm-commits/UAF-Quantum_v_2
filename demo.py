import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =============================================
# UAF-Quantum v2: Двухуровневая система (стабильная)
# =============================================

hbar = 1.0

def uaf_two_level(t, y, Omega=1.2, Delta=0.4, lam=0.08, eta=1.1):
    Gamma, alpha, phi = y
    
    # Ограничения
    Gamma = np.clip(Gamma, 1e-5, 1.0)
    alpha = np.clip(alpha, 0.0, 1.0)
    
    Omega_eff = Omega * (1 - alpha)**1.5   # сильное влияние декогеренции
    
    # Уравнения
    dGamma_dt = (eta * (1 - alpha)**2 * Gamma * np.cos(phi) 
                 - lam * alpha * Gamma) / hbar
    
    dalpha_dt = (0.04 * alpha * (1 - alpha) 
                 + 0.22 * (1 - alpha) * (1 - Gamma**2)) / hbar
    
    dphi_dt = Delta + Omega_eff * Gamma
    
    return [dGamma_dt, dalpha_dt, dphi_dt]


# Начальные условия
y0 = [0.95, 0.03, 0.0]

t_span = (0, 150)
t_eval = np.linspace(0, 150, 2000)

sol = solve_ivp(uaf_two_level, t_span, y0, t_eval=t_eval, 
                method='LSODA', rtol=1e-7, atol=1e-8)

# Постобработка
Gamma = np.clip(sol.y[0], 0, 1)
alpha = np.clip(sol.y[1], 0, 1)
phi = sol.y[2]

P_ex = 0.5 * (1 + Gamma * np.cos(phi))   # Примерная вероятность

# ====================== Графики ======================
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.plot(sol.t, Gamma, 'b-', lw=2.5, label=r'$\Gamma$ — Когерентность (t_self)')
plt.plot(sol.t, alpha, 'r-', lw=2.5, label=r'$\alpha$ — Интеграция в прошлое')
plt.xlabel(r'Внутреннее время $\tau$')
plt.ylabel('Значение')
plt.title('UAF-Q: Декогеренция двухуровневой системы')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(sol.t, P_ex, 'g-', lw=2, label='Населённость верхнего уровня (примерно)')
plt.xlabel(r'Внутреннее время $\tau$')
plt.ylabel('Вероятность')
plt.title('Динамика населённостей')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(sol.t, Gamma * np.cos(phi), 'purple', lw=2, label='Re')
plt.plot(sol.t, Gamma * np.sin(phi), 'orange', lw=2, label='Im')
plt.xlabel(r'Внутреннее время $\tau$')
plt.ylabel('Когерентная амплитуда')
plt.title('Осцилляции когерентности')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(Gamma, alpha, 'k-', lw=2, label='Траектория')
plt.xlabel(r'$\Gamma$')
plt.ylabel(r'$\alpha$')
plt.title('Фазовый портрет (Γ → α)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Вывод результатов
print("=== Результат симуляции UAF-Q (двухуровневая) ===")
print(f"Финальная когерентность Γ     : {Gamma[-1]:.4f}")
print(f"Финальная интеграция α        : {alpha[-1]:.4f}")
print(f"Средняя когерентность         : {np.mean(Gamma):.4f}")
print(f"Максимальное значение α       : {alpha.max():.4f}")
