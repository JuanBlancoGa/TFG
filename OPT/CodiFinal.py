from pyomo.environ import *
import pandas as pd
import matplotlib.pyplot as plt
from pyomo.opt import SolverFactory

# ==========================
# Seleccion Modelo
# ==========================
#EST=(1- Alta Entrada 2-Media Entrada  3-Baja entrada)
#y= (0- Sin demanda 1- Con Demanda)
EST=3
y_value=1

# ==========================
# 0. Cargar datos 
# ==========================

if EST==1:
    df = pd.read_excel("Libro_AltaEntrada.xlsx")
    EcoMax=652
    EcoMin=34.97
    EcoMax2=485.27
    EcoMin2=13.95
    QPujada=137
    QBaixada=104
    QPujada2=135
    QBaixada2=102.94
    
elif EST==2:
    df = pd.read_excel("Libro_MediaEntrada.xlsx")
    EcoMax=365.44
    EcoMin=39.39
    EcoMax2=259.42
    EcoMin2=19.55
    QPujada=126
    QBaixada=98
    QPujada2=120
    QBaixada2=96
else:
    df = pd.read_excel("Libro_BajaEntrada.xlsx")
    EcoMax=131.89
    EcoMin=27.32
    EcoMax2=52.17
    EcoMin2=7.74
    QPujada=96
    QBaixada=78
    QPujada2=66.65
    QBaixada2=59.24


df = df.dropna(subset=['t', 'Qe_1', 'Qe_3'])
df['t'] = df['t'].astype(int)
N = 195
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


sigma_const = {
    1: eta_gen * eta_tur * rho * g * 104,
    2: eta_gen * eta_tur * rho * g * 18,
    3: eta_gen * eta_tur * rho * g * 90,
}

model.sigma = Param(model.I, initialize=sigma_const)
h_max_data = {1: 229, 2: 130, 3: 194.24}
A_em_data = {1: 14620000, 2: 560000, 3: 5350000}
h0 = {1: 222, 2: 127.5, 3: 190}
Q_max_data = {(1, 1): 500, (2, 1): 375, (3, 1): 223}
P_max_data = {(1, 1): 425, (2, 1): 60.1, (3, 1): 180.8}
h_min_data = {1: 217, 2: 125, 3: 180}
h_al_data = {1: 214, 2: 123, 3: 177}

model.h_min = Param(model.I, initialize=h_min_data)
model.h_max = Param(model.I, initialize=h_max_data)
model.h_al = Param(model.I, initialize=h_al_data)
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
    entrada = m.Qe_1[t] if i == 1 else m.Qe_2[t] if i == 2 else m.Qe_3[t]
    return m.h[i, t + 1] == m.h[i, t] + (T_value / m.A_em[i]) * (
        entrada - sum(m.Q[i, j, t] for j in m.J_all)
    ) + m.p[i, t]/1000
model.h_evolution = Constraint(model.I, model.T, rule=h_evolution_rule)



# ==========================
# 10. Restricciones de generación binaria
# ==========================
M = 1e5
model.no_gen_if_low_h = Constraint(model.I, model.J, model.T, rule=lambda m, i, j, t: m.h[i, t] >= m.h_min[i] - (1 - m.z[i, j, t]) * M)
model.enforce_z_limit = Constraint(model.I, model.J, model.T, rule=lambda m, i, j, t: m.Q[i, j, t] <= m.Q_max[i, j] * m.z[i, j, t])

# ==========================
# 11. Otras restricciones
# ==========================
model.volume_cap = Constraint(model.I, model.T, rule=lambda m, i, t:  m.h[i, t] <= m.h_max[i])
model.volume_min = Constraint(model.I, model.T, rule=lambda m, i, t:  m.h_al[i]<= m.h[i, t]  )
model.max_flow = Constraint(model.I, model.J, model.T, rule=lambda m, i, j, t: m.Q[i, j, t] <= m.Q_max[i, j])
model.max_power = Constraint(model.I, model.J, model.T, rule=lambda m, i, j, t: m.sigma[i] * m.Q[i, j, t] <= m.P_max[i, j])


model.flow_reg_up = Constraint(model.T, rule=lambda m, t: Constraint.Skip if t == IT else sum(m.Q[2, j, t + 1] for j in m.J_all) - sum(m.Q[2, j, t] for j in m.J_all) <= QBaixada)
model.flow_reg_down = Constraint(model.T, rule=lambda m, t: Constraint.Skip if t == IT else sum(m.Q[2, j, t] for j in m.J_all) - sum(m.Q[2, j, t + 1] for j in m.J_all) <= QBaixada2)

model.flow_reg_up3 = Constraint(model.T, rule=lambda m, t: Constraint.Skip if t == IT else sum(m.Q[3, j, t + 1] for j in m.J_all) - sum(m.Q[3, j, t] for j in m.J_all) <= QPujada)
model.flow_reg_down3 = Constraint(model.T, rule=lambda m, t: Constraint.Skip if t == IT else sum(m.Q[3, j, t] for j in m.J_all) - sum(m.Q[3, j, t + 1] for j in m.J_all) <= QPujada2)


model.caudal_central2_limit = Constraint(model.T, rule=lambda m, t: sum(m.Q[2, j, t] for j in m.J_all) <= EcoMax)
model.caudal_central3_limit = Constraint(model.T, rule=lambda m, t: sum(m.Q[3, j, t] for j in m.J_all) <= EcoMax2)

model.caudal_central2eco_limit = Constraint(model.T, rule=lambda m, t: sum(m.Q[2, j, t] for j in m.J_all) >= EcoMin)
model.caudal_central3eco_limit = Constraint(model.T, rule=lambda m, t: sum(m.Q[3, j, t] for j in m.J_all) >= EcoMin2)



model.demand_no_supplied = Var(model.T, domain=NonNegativeReals)

def unmet_demand_rule(m, t):
    generation = sum(m.sigma[i] * m.Q[i, j, t] for i in m.I for j in m.J)
    return generation + m.demand_no_supplied[t] >= m.D[t]*y_value
model.unmet_demand_constraint = Constraint(model.T, rule=unmet_demand_rule)

# ==========================
# 12. Función objetivo
# ==========================
model.obj = Objective(
    rule=lambda m: sum(
        (
            sum(m.sigma[i] * m.Q[i, j, t] for i in m.I for j in m.J)  # j=1: generació
            
        ) * m.c[t]
        - sum(0.0001 * m.Q[i, 2, t] for i in m.I) - 1000* m.demand_no_supplied[t]*y_value # penalització j=2 (sobreeixidor)
        for t in m.T
    ),
    sense=maximize
)

# ==========================
# 13. Resolver
# ==========================
solver = SolverFactory('glpk', executable="C:/Users/juaaa/Desktop/Upc/TFG/PROYECTO/Utils/CODIS/Optisimfinal/glpk-4.65/w64/glpsol.exe")
solver.options['mipgap'] = 0.01
results = solver.solve(model, tee=True)


# ==========================
# 14. Gráficas
# ==========================
# Niveles (sin cambios)
for i in model.I:
    h_min_i = value(model.h_min[i])
    h_max_i = value(model.h_max[i])
    h_al_i = value(model.h_al[i])
    plt.figure(figsize=(12, 6))
    h_vals = [value(model.h[i, t]) for t in model.T]
    plt.plot(model.T, h_vals, label=f"h_{i}", color='blue')
    plt.axhline(h_min_i, color='orange', linestyle='--', label="h mínima tècnica")
    plt.axhline(h_max_i, color='green', linestyle='--', label="h màxima")
    plt.axhline(h_al_i, color='red', linestyle='--', label="h mínima d'operació")
    plt.title(f"Nivell central {i}")
    plt.xlabel("Temps (t)")
    plt.ylabel("Altura (msnm)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Caudales (con nuevas leyendas)
for i in model.I:
    plt.figure(figsize=(12, 6))
    for j in model.J_all:
        etiqueta = "Caudal generació" if j == 1 else "Caudal sobreeixidor"
        plt.plot(model.T, [value(model.Q[i, j, t]) for t in model.T], label=etiqueta)
    plt.title(f"Cabal central {i}")
    plt.xlabel("Temps (t)")
    plt.ylabel("Cabal (m³/s)")
    plt.legend()
    plt.grid(True)
    plt.show()

# ==========================
# 15. Gráfica: Caudal Total vs Precio
# ==========================
tiempos = list(model.T)
caudal_total = [sum(value(model.Q[i, j, t]) for i in model.I for j in model.J) for t in tiempos]
precio_electricidad = [value(model.c[t]) for t in tiempos]

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.set_xlabel('Temps (t)')
ax1.set_ylabel('Cabal total de generació (m³/s)', color='tab:blue')
ax1.plot(tiempos, caudal_total, color='tab:blue', label="Cabal total")
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Preu Marginal', color='tab:red')
ax2.plot(tiempos, precio_electricidad, color='tab:red', linestyle='--', label="Preu")
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title("Cabal total de generació vs Preu Marginal")
fig.tight_layout()
plt.grid(True)
plt.show()

# ==========================
# 16. Comparación: Generació vs Demanda
# ==========================
potencia_generada = [sum(value(model.sigma[i]) * value(model.Q[i, j, t]) for i in model.I for j in model.J) for t in model.T]
demanda = [value(model.D[t]) for t in model.T]

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


temps = list(model.T)
generacio = [
    sum(value(model.sigma[i]) * value(model.Q[i, j, t]) for i in model.I for j in model.J)
    for t in model.T
]
demanda = [value(model.D[t]) for t in model.T]
diferencia = [g - d for g, d in zip(generacio, demanda)]

# Calcular la demanda no abastida total
demanda_no_abastida = sum(d - g for g, d in zip(generacio, demanda) if g < d)

plt.figure(figsize=(12, 6))
plt.plot(temps, diferencia, marker='o', linestyle='-', color='green', label='Generació - Demanda')
plt.axhline(0, color='black', linestyle='--')

print(demanda_no_abastida)

# Mostrar la suma total al gràfic
plt.xlabel("Temps (t)")
plt.ylabel("Diferència (MW)")
plt.title("Generació - Demanda amb càlcul de demanda no abastida")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
