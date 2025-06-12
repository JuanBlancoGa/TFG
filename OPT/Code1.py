from pyomo.environ import *
import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# 0. Cargar datos desde Excel
# ==========================
df = pd.read_excel("entrada_modelo_semana1.xlsx")
N = 48
df = df[df["t"] <= N].copy()

# Crear diccionarios para Pyomo
D_data = df.set_index('t')['D'].to_dict()
c_data = df.set_index('t')['c'].to_dict()

Qe_data = {(1, t): q for t, q in df[['t', 'Qe_1']].values}
Qe_data.update({(3, t): q for t, q in df[['t', 'Qe_3']].values})

# 🔧 Inicializar Qe[2, t] con 0.0 (será sobrescrito por restricción)

p_data = {(1, t): p for t, p in zip(df['t'], df['p_1'])}
p_data.update({(2, t): p for t, p in zip(df['t'], df['p_2'])})
p_data.update({(3, t): p for t, p in zip(df['t'], df['p_3'])})

# ==========================
# 1. Crear el modelo
# ==========================
model = ConcreteModel()

# ==========================
# 2. Parámetros globales
# ==========================
C = 3
NT_plus_1 = 2
NT_max = NT_plus_1 - 1
IT = df['t'].max()
T_value = 1
y_value = 0
beta = 15

# ==========================
# 3. Conjuntos
# ==========================
model.T = RangeSet(1, IT)
model.I = RangeSet(1, C)
model.J = RangeSet(1, NT_max)
model.J_all = RangeSet(1, NT_plus_1)

# ==========================
# 4. Parámetros físicos
# ==========================
eta_gen = 0.95
eta_tur = 0.90
rho = 1000
g = 9.81 * 1e-6  # MW·s/kg·m

# Constantes sigma por central (h_1=95, h_2=17, h_3=40)
sigma_const = {
    1: eta_gen * eta_tur * rho * g * 95,
    2: eta_gen * eta_tur * rho * g * 17,
    3: eta_gen * eta_tur * rho * g * 40,
}
model.sigma = Param(model.I, initialize=sigma_const)

# ==========================
# 5. Parámetros fijos
# ==========================
h_min_data = {1: 95, 2: 15, 3: 38}
V_max_data = {1: 1447380000, 2: 13000000, 3: 230050000}
A_em_data = {1: 14620000, 2: 560000, 3: 5350000}
h0 = {1: 98, 2: 17, 3: 40}

Q_max_data = {
    (1, 1): 428 / sigma_const[1],
    (2, 1): 59.9 / sigma_const[2],
    (3, 1): 179.9 / sigma_const[3],
}

P_max_data = {
    (1, 1): 428,
    (2, 1): 59.9,
    (3, 1): 179.9
}

Q_max_s = 100000

model.h_min = Param(model.I, initialize=h_min_data)
model.V_max = Param(model.I, initialize=V_max_data)
model.A_em = Param(model.I, initialize=A_em_data)
model.Q_max = Param(model.I, model.J, initialize=Q_max_data)
model.P_max = Param(model.I, model.J, initialize=P_max_data)

# ==========================
# 6. Parámetros desde Excel
# ==========================
model.D = Param(model.T, initialize=D_data)
model.c = Param(model.T, initialize=c_data)
model.Qe = Param(model.I, model.T, initialize=Qe_data, mutable=True)
model.p = Param(model.I, model.T, initialize=p_data)

# ==========================
# 7. Variables
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
# 9. Evolución de h
# ==========================
def h_evolution_rule(m, i, t):
    if t == IT:
        return Constraint.Skip
    return m.h[i, t + 1] == m.h[i, t] + (T_value * 3600 / m.A_em[i]) * (
        m.Qe[i, t] - sum(m.Q[i, j, t] for j in m.J_all)
    ) + m.p[i, t]
model.h_evolution = Constraint(model.I, model.T, rule=h_evolution_rule)


# ==========================
# 10. Restricciones con binaria
# ==========================
M = 1e5
model.no_gen_if_low_h = Constraint(model.I, model.J, model.T,
    rule=lambda m, i, j, t: m.h[i, t] >= m.h_min[i] - (1 - m.z[i, j, t]) * M)
model.enforce_z_limit = Constraint(model.I, model.J, model.T,
    rule=lambda m, i, j, t: m.Q[i, j, t] <= m.Q_max[i, j] * m.z[i, j, t])

# ==========================
# 11. Otras restricciones
# ==========================
#model.volume_cap = Constraint(model.I, model.T, rule=lambda m, i, t: m.h[i, t] <= m.V_max[i] / m.A_em[i])
model.max_flow = Constraint(model.I, model.J, model.T, rule=lambda m, i, j, t: m.Q[i, j, t] <= m.Q_max[i, j])
model.max_power = Constraint(model.I, model.J, model.T, rule=lambda m, i, j, t: m.sigma[i] * m.Q[i, j, t] <= m.P_max[i, j])
model.linked_flow = Constraint(model.T, rule=lambda m, t: m.Qe[2, t] == sum(m.Q[1, j, t] for j in m.J_all))
model.outflow_limit = Constraint(model.T, rule=lambda m, t: sum(m.Q[2, j, t] for j in m.J_all) + sum(m.Q[3, j, t] for j in m.J_all) <= Q_max_s)

'''def flow_regulation_upper(m, t):
    if t == IT: return Constraint.Skip
    return sum(m.Q[2, j, t + 1] for j in m.J_all) - sum(m.Q[2, j, t] for j in m.J_all) <= beta * sum(m.Q[2, j, t] for j in m.J_all)

def flow_regulation_lower(m, t):
    if t == IT: return Constraint.Skip
    return sum(m.Q[2, j, t] for j in m.J_all) - sum(m.Q[2, j, t + 1] for j in m.J_all) <= beta * sum(m.Q[2, j, t] for j in m.J_all)

model.flow_reg_up = Constraint(model.T, rule=flow_regulation_upper)
model.flow_reg_down = Constraint(model.T, rule=flow_regulation_lower)'''

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
    ) - 0.01 * sum(m.h[i, t] for i in m.I for t in m.T)

model.obj = Objective(rule=objective_rule, sense=maximize)

# ==========================
# 13. Resolver
# ==========================
solver = SolverFactory('glpk', executable="C:/Users/juaaa/Desktop/Upc/TFG/PROYECTO/Utils/CODIS/Optisimfinal/glpk-4.65/w64/glpsol.exe")
solver.options['mipgap'] = 0.05 
results = solver.solve(model, tee=True)
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
    h_max_i = value(model.V_max[i]) / value(model.A_em[i])

    plt.figure(figsize=(12, 4))
    h_vals = [value(model.h[i, t]) for t in model.T]
    plt.plot(model.T, h_vals, label=f"h_{i}", color='blue')
    plt.axhline(h_min_i, color='red', linestyle='--', label="h mínima")
    plt.axhline(h_max_i, color='green', linestyle='--', label="h máxima")
    plt.title(f"Nivel de embalse central {i}")
    plt.xlabel("Tiempo (t)")
    plt.ylabel("Altura (m)")
    plt.legend()
    plt.grid(True)
    plt.show()

for i in model.I:
    plt.figure(figsize=(12, 4))
    for j in model.J_all:
        plt.plot(model.T, [value(model.Q[i, j, t]) for t in model.T], label=f"Q_{i},{j}")
    plt.title(f"Caudales central {i}")
    plt.xlabel("Tiempo (t)")
    plt.ylabel("Caudal (m³/s)")
    plt.legend()
    plt.grid(True)
    plt.show()
