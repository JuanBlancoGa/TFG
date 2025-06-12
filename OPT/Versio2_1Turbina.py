from pyomo.environ import *
import pandas as pd
import matplotlib.pyplot as plt
from pyomo.opt import SolverFactory


# ==========================
# 0. Cargar datos desde Excel
# ==========================
df = pd.read_excel("Libro_MediaEntrada")
df = pd.read_excel("Libro_BajaEntrada")
df = pd.read_excel("Libro_AltaEntrada")
df = df.dropna(subset=['t', 'Qe_1', 'Qe_3'])
df['t'] = df['t'].astype(int)
N = 96
df = df[df['t'] <= N].copy()

# ==========================
# 1. Crear diccionarios desde Excel
# ==========================
Qe_data = {(1, row['t']): row['Qe_1'] for _, row in df.iterrows()}
Qe_data.update({(3, row['t']): row['Qe_3'] for _, row in df.iterrows()})
D_data = df.set_index("t")['D'].to_dict()
c_data = df.set_index("t")['c'].to_dict()

p_data = {
    (1, t): p for t, p in zip(df['t'], df['p_1'])
}
p_data.update({
    (2, t): p for t, p in zip(df['t'], df['p_2'])
})
p_data.update({
    (3, t): p for t, p in zip(df['t'], df['p_3'])
})

# ==========================
# 2. Crear el modelo
# ==========================
model = ConcreteModel()

# Parámetros globales
C = 3
NT_plus_1 = 2
NT_max = NT_plus_1 - 1
IT = df['t'].max()
model.T = RangeSet(1, IT)
model.I = RangeSet(1, C)
model.J = RangeSet(1, NT_max)
model.J_all = RangeSet(1, NT_plus_1)


# ==========================
# 3. Parámetros 
# ==========================
eta_gen = 0.95
eta_tur = 0.90
rho = 1000
g = 9.81 * 1e-6
T_value = 3600
y_value = 0

# Calcular sigmas constantes
sigma_const = {
    1: eta_gen * eta_tur * rho * g * 104,
    2: eta_gen * eta_tur * rho * g * 18,
    3: eta_gen * eta_tur * rho * g * 90,
}


model.sigma = Param(model.I, initialize=sigma_const)
h_max_data = {1: 229, 2: 130, 3: 194.24}
A_em_data = {1: 14620000, 2: 560000, 3: 5350000}
h0 = {1: 222, 2: 127.5, 3: 190}


Q_max_data = {
    (1, 1): 500, 
    (2, 1): 375, 
    (3, 1): 223, 
}

P_max_data = {
    (1, 1): 425, 
    (2, 1): 60.1,
    (3, 1): 180.8, 
}


h_min_data = {1: 217, 2: 125, 3: 185}


model.h_min = Param(model.I, initialize=h_min_data)
model.h_max = Param(model.I, initialize=h_max_data)
model.A_em = Param(model.I, initialize=A_em_data)
model.Q_max = Param(model.I, model.J, initialize=Q_max_data)
model.P_max = Param(model.I, model.J, initialize=P_max_data)

model.D = Param(model.T, initialize=D_data)
model.c = Param(model.T, initialize=c_data)
model.p = Param(model.I, model.T, initialize=p_data)
model.Qe_1 = Param(model.T, initialize={t: val for (i, t), val in Qe_data.items() if i == 1}, mutable=True)
model.Qe_3 = Param(model.T, initialize={t: val for (i, t), val in Qe_data.items() if i == 3}, mutable=True)


# ==========================
# 4. Variables
# ==========================
model.Q = Var(model.I, model.J_all, model.T, domain=NonNegativeReals)
model.h = Var(model.I, model.T, domain=NonNegativeReals)
model.z = Var(model.I, model.J, model.T, domain=Binary)

# ==========================
# 8. Altura inicial
# ==========================
def h_initial_rule(m, i):
    return m.h[i, 1] == h0[i]
model.h_init = Constraint(model.I, rule=h_initial_rule)

# ==========================
# 9. Orden Fluido
# ==========================
def linked_flow_rule(m, t):
    return m.Qe_2[t] == sum(m.Q[1, j, t] for j in m.J_all)
model.Qe_2 = Var(model.T, domain=NonNegativeReals)
model.linked_flow = Constraint(model.T, rule=linked_flow_rule)

# ==========================
# 9. Evolución de h
# ==========================
def h_evolution_rule(m, i, t):
    if t == IT:
        return Constraint.Skip
    if i == 1:
        return m.h[i, t + 1] == m.h[i, t] + (T_value / m.A_em[i]) * (
            m.Qe_1[t] - sum(m.Q[i, j, t] for j in m.J_all)
        ) + m.p[i, t]/1000
    elif i == 2:
        return m.h[i, t + 1] == m.h[i, t] + (T_value / m.A_em[i]) * (
            m.Qe_2[t] - sum(m.Q[i, j, t] for j in m.J_all)
        ) + m.p[i, t]/1000
    elif i == 3:
        return m.h[i, t + 1] == m.h[i, t] + (T_value / m.A_em[i]) * (
            m.Qe_3[t] - sum(m.Q[i, j, t] for j in m.J_all)
        ) + m.p[i, t]/1000
model.h_evolution = Constraint(model.I, model.T, rule=h_evolution_rule)

# ==========================
# 10. Restricciones de generación binaria
# ==========================
M = 1e5
def min_height_no_generation_rule(m, i, j, t):
    return m.h[i, t] >= m.h_min[i] - (1 - m.z[i, j, t]) * M
model.no_gen_if_low_h = Constraint(model.I, model.J, model.T, rule=min_height_no_generation_rule)

def enforce_generation_only_if_allowed(m, i, j, t):
    return m.Q[i, j, t] <= m.Q_max[i, j] * m.z[i, j, t]
model.enforce_z_limit = Constraint(model.I, model.J, model.T, rule=enforce_generation_only_if_allowed)


# ==========================
# 11. Otras restricciones
# ==========================
model.volume_cap = Constraint(model.I, model.T, rule=lambda m, i, t: m.h[i, t] <= m.h_max[i])
model.max_flow = Constraint(model.I, model.J, model.T, rule=lambda m, i, j, t: m.Q[i, j, t] <= m.Q_max[i, j])
model.max_power = Constraint(model.I, model.J, model.T, rule=lambda m, i, j, t: m.sigma[i] * m.Q[i, j, t] <= m.P_max[i, j])


def flow_regulation_upper(m, t):
    if t == IT: return Constraint.Skip
    #return sum(m.Q[2, j, t + 1] for j in m.J_all) - sum(m.Q[2, j, t] for j in m.J_all) <= beta * sum(m.Q[2, j, t] for j in m.J_all)
    return sum(m.Q[2, j, t + 1] for j in m.J_all) - sum(m.Q[2, j, t] for j in m.J_all) <= 96

def flow_regulation_lower(m, t):
    if t == IT: return Constraint.Skip
    #return sum(m.Q[2, j, t] for j in m.J_all) - sum(m.Q[2, j, t + 1] for j in m.J_all) <= beta * sum(m.Q[2, j, t] for j in m.J_all)
    return sum(m.Q[2, j, t] for j in m.J_all) - sum(m.Q[2, j, t + 1] for j in m.J_all) <= 78


model.flow_reg_up = Constraint(model.T, rule=flow_regulation_upper)
model.flow_reg_down = Constraint(model.T, rule=flow_regulation_lower)

def flow_regulation_upper3(m, t):
    if t == IT: return Constraint.Skip
    #return sum(m.Q[3, j, t + 1] for j in m.J_all) - sum(m.Q[3, j, t] for j in m.J_all) <= beta * sum(m.Q[3, j, t] for j in m.J_all)
    return sum(m.Q[3, j, t + 1] for j in m.J_all) - sum(m.Q[3, j, t] for j in m.J_all) <= 66.65

def flow_regulation_lower3(m, t):
    if t == IT: return Constraint.Skip
    #return sum(m.Q[2, j, t] for j in m.J_all) - sum(m.Q[2, j, t + 1] for j in m.J_all) <= beta * sum(m.Q[3, j, t] for j in m.J_all)
    return sum(m.Q[3, j, t] for j in m.J_all) - sum(m.Q[3, j, t + 1] for j in m.J_all) <= 59.24


model.flow_reg_up3 = Constraint(model.T, rule=flow_regulation_upper3)
model.flow_reg_down3 = Constraint(model.T, rule=flow_regulation_lower3)

def caudal_central2_ecomax_rule(m, t):
    return sum(m.Q[2, j, t] for j in m.J_all) <= 131.89
model.caudal_central2_limit = Constraint(model.T, rule=caudal_central2_ecomax_rule)

def caudal_central3_ecomax_rule(m, t):
    return sum(m.Q[3, j, t] for j in m.J_all) <= 52.17
model.caudal_central3_limit = Constraint(model.T, rule=caudal_central2_ecomax_rule)

def caudal_central2_ecomin_rule(m, t):
    return sum(m.Q[2, j, t] for j in m.J_all) >= 28
model.caudal_central2eco_limit = Constraint(model.T, rule=caudal_central2_ecomin_rule)

def caudal_central3_ecomin_rule(m, t):
    return sum(m.Q[3, j, t] for j in m.J_all) >= 8
model.caudal_central3eco_limit = Constraint(model.T, rule=caudal_central3_ecomin_rule)

# ==========================
# 12. Función objetivo
# ==========================
def objective_rule(m):
    return sum(
        (
            sum(m.sigma[i] * m.Q[i, j, t] * T_value for i in m.I for j in m.J)
            - y_value * m.D[t]
        ) * m.c[t]
        for t in m.T
    ) 

model.obj = Objective(rule=objective_rule, sense=maximize)

# ==========================
# 13. Resolver
# ==========================
solver = SolverFactory('glpk', executable="C:/Users/juaaa/Desktop/Upc/TFG/PROYECTO/Utils/CODIS/Optisimfinal/glpk-4.65/w64/glpsol.exe")
solver.options['mipgap'] = 0.01

#solver = SolverFactory("highs_cmd", executable="C:/Users/juaaa/Desktop/Upc/TFG/PROYECTO/Utils/CODIS/Optisimfinal/bin/highs.exe")

results = solver.solve(model, tee=True)

#results = solver.solve(model, logfile="highs_log.txt")

if (results.solver.termination_condition != TerminationCondition.optimal):
    print("⚠️ Modelo infactible")

    print("\n🔍 Estado inicial h0 vs h_min:")
    for i in model.I:
        print(f"Central {i}: h0 = {h0[i]} | h_min = {h_min_data[i]}")

    print("\n🔍 Potencia vs caudal permitido:")
    for (i, j), qmax in Q_max_data.items():
        if qmax > 0:
            potencia = sigma_const[i] * qmax
            pmax = P_max_data[(i, j)]
            if potencia > pmax:
                print(f"⚠️ ({i},{j}): σ·Q_max = {potencia:.2f} > P_max = {pmax}")

for i in model.I:
    total_qmax = sum(
        Q_max_data.get((i, j), 0)  # usa 0 si no existe
        for j in model.J_all
    )
    print(f"Central {i}: Q_max total (incluyendo j=5 si existe) = {total_qmax:.2f} m³/s")

# ==========================
# 14. Gráficas
# ==========================
for i in model.I:
    h_min_i = value(model.h_min[i])
    h_max_i = value(model.h_max[i]) 

    plt.figure(figsize=(12, 4))
    h_vals = [value(model.h[i, t]) for t in model.T]
    plt.plot(model.T, h_vals, label=f"h_{i}", color='blue')
    plt.axhline(h_min_i, color='red', linestyle='--', label="h mínima")
    plt.axhline(h_max_i, color='green', linestyle='--', label="h màxima")
    plt.title(f"Nivell central {i}")
    plt.xlabel("Temps (t)")
    plt.ylabel("Altura (msnm)")
    plt.legend()
    plt.grid(True)
    plt.show()

for i in model.I:
    plt.figure(figsize=(12, 4))
    for j in model.J_all:
        plt.plot(model.T, [value(model.Q[i, j, t]) for t in model.T], label=f"Q_{i},{j}")
    plt.title(f"Cabal central {i}")
    plt.xlabel("Temps (t)")
    plt.ylabel("Cabal (m³/s)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
#CAUDALES TOTALES DE SALIDA
'''plt.plot(model.T, [value(model.Q[2, 1, t]+model.Q[2, 2, t]) for t in model.T], label=f"TOT")
plt.title(f"Caudales central {i}")
plt.xlabel("Tiempo (t)")
plt.ylabel("Caudal (m³/s)")
plt.legend()
plt.grid(True)
plt.show()
plt.plot(model.T, [value(model.Q[3, 1, t]+model.Q[3, 2, t]) for t in model.T], label=f"TOT")
plt.title(f"Caudales central {i}")
plt.xlabel("Tiempo (t)")
plt.ylabel("Caudal (m³/s)")
plt.legend()
plt.grid(True)
plt.show()'''

# ==========================
# 15. Gráfica de caudales de entrada
# ==========================
'''plt.figure(figsize=(12, 5))
plt.plot(model.T, [value(model.Qe_1[t]) for t in model.T], label='Qe_1 (Central 1)', linestyle='-', marker='o')
plt.plot(model.T, [value(model.Qe_2[t]) for t in model.T], label='Qe_2 (Central 2)', linestyle='--', marker='x')
plt.plot(model.T, [value(model.Qe_3[t]) for t in model.T], label='Qe_3 (Central 3)', linestyle='-.', marker='s')
plt.title("Caudales de entrada por central")
plt.xlabel("Tiempo (t)")
plt.ylabel("Caudal de entrada (m³/s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()'''

resultados = []

for i in model.I:
    for t in model.T:
        for j in model.J_all:
            resultados.append({
                'Central': i,
                'Turbina': j,
                'Temps': t,
                'Q (m³/s)': value(model.Q[i, j, t]),
                'z': value(model.z[i, j, t]) if j in model.J else None
            })

        # Caudales de entrada
        resultados.append({
            'Central': i,
            'Turbina': 'Qe',
            'Temps': t,
            'Q (m³/s)': (
                value(model.Qe_1[t]) if i == 1 else
                value(model.Qe_2[t]) if i == 2 else
                value(model.Qe_3[t]) if i == 3 else None
            ),
            'z': None
        })

        # Altura del embalse
        resultados.append({
            'Central': i,
            'Turbina': 'h',
            'Temps': t,
            'Q (m³/s)': value(model.h[i, t]),
            'z': None
        })

# Crear DataFrame
df_resultados = pd.DataFrame(resultados)
df_resultados.to_excel("resultados_optimizacion.xlsx", index=False)

import matplotlib.pyplot as plt

# Extraer datos
tiempos = list(model.T)
caudal_total = [
    sum(value(model.Q[i, j, t]) for i in model.I for j in model.J)
    for t in tiempos
]
precio_electricidad = [value(model.c[t]) for t in tiempos]

# Crear gráfico
fig, ax1 = plt.subplots(figsize=(12, 6))

# Eje izquierdo: caudal
color1 = 'tab:blue'
ax1.set_xlabel('Temps (t)')
ax1.set_ylabel('Cabal total de generació (m³/s)', color=color1)
ax1.plot(tiempos, caudal_total, color=color1, label="Cabal total")
ax1.tick_params(axis='y', labelcolor=color1)

# Eje derecho: precio
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Preu Marginal', color=color2)
ax2.plot(tiempos, precio_electricidad, color=color2, linestyle='--', label="Preu")
ax2.tick_params(axis='y', labelcolor=color2)

# Título y leyenda
plt.title("Cabal total de generació vs Preu Marginal")
fig.tight_layout()
plt.grid(True)
plt.show()

# ==========================
# Comparación: Demanda vs Potencia Generada
# ==========================

potencia_generada = []
demanda = []

for t in model.T:
    # Potencia total generada en t (MW)
    p_total = sum(value(model.sigma[i]) * value(model.Q[i, j, t]) for i in model.I for j in model.J)
    potencia_generada.append(p_total)
    demanda.append(value(model.D[t]))

# Graficar comparación
plt.figure(figsize=(12, 6))
plt.plot(model.T, potencia_generada, label='Generació (MW)', marker='o')
plt.plot(model.T, demanda, label='Demanda (MW)', marker='x', linestyle='--')
plt.title("Comparació Generació vs Demanda")
plt.xlabel("Temps (t)")
plt.ylabel("Potència (MW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


