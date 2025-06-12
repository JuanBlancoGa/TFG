import numpy as np

# FUNCIO PER AFEGIR A LA MATRIU
def afegir_linia(Ybus, i, j, R, X):
    """
    Afegeix una línia entre els busos i i j a la matriu Ybus.

    Parameters:
    Ybus : ndarray
        Matriu d’admitàncies (complexa) de mida [nbus x nbus]
    i, j : int
        Índexs dels busos (comencen a 0)
    R, X : float
        Resistència i reactància de la línia (Ω)
    """
    Y = 1 / complex(R, X)
    Ybus[i, j] -= Y
    Ybus[j, i] -= Y
    Ybus[i, i] += Y
    Ybus[j, j] += Y

# Constants
tolerance = 1e-6
max_iter = 20

# Nombre de busos
nbus = 9

# Classificació de busos
slack = 0
pq_buses = [1, 4, 6, 7]
pv_buses = [2, 3, 5, 8]
pv_pq = pq_buses + pv_buses

# Inicialització de tensions i angles
V = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,1.0,1.0])  
delta = np.radians(np.zeros(nbus))  

# Potències especificades (per unitat) Orde dels Busos
P_spec = np.array([0.0, -0.5556, 0.7025, 0.5147, -0, +0.1669,-0.2778 ,0 ,+0.5022 ])
Q_spec = np.array([0.0, -0.2778, 0.0   , 0     ,  0, 0.0    ,-0.1389 ,0 , 0.0    ])  # només per PQ

# Matriu Ybus (9x9)
Ybus = np.zeros((nbus, nbus), dtype=complex)

afegir_linia(Ybus,0,1,0.0123,0.0846)
afegir_linia(Ybus,1,2,1e-6,0.1400)
afegir_linia(Ybus,1,3,1e-6,0.2057)
afegir_linia(Ybus,1,4,0.0032,0.0219)
afegir_linia(Ybus,4,5,1e-6,0.7200)
afegir_linia(Ybus,4,6,0.004,0.2571)
afegir_linia(Ybus,6,7,0.0031,0.0211)
afegir_linia(Ybus,7,8,0.00,0.2571)

# Iteració Newton-Raphson
for iteration in range(100):

    # Trobar valors de P i Q
    P_calc = np.zeros(nbus)
    Q_calc = np.zeros(nbus)
    
    for i in range(nbus):
        for j in range(nbus):
            G = Ybus[i, j].real
            B = Ybus[i, j].imag
            angle = delta[i] - delta[j]
            P_calc[i] += V[i] * V[j] * (G * np.cos(angle) + B * np.sin(angle))
            Q_calc[i] += V[i] * V[j] * (G * np.sin(angle) - B * np.cos(angle))
    
    dP = P_spec[pv_pq] - P_calc[pv_pq]
    dQ = Q_spec[pq_buses] - Q_calc[pq_buses]
    mismatch = np.concatenate([dP, dQ])

    if np.max(np.abs(mismatch)) < tolerance:
        break

    # Jacobiana
    n_pv_pq = len(pv_pq)
    n_pq = len(pq_buses)
    J1 = np.zeros((n_pv_pq, n_pv_pq))
    J2 = np.zeros((n_pv_pq, n_pq))
    J3 = np.zeros((n_pq, n_pv_pq))
    J4 = np.zeros((n_pq, n_pq))

    for i_idx, i in enumerate(pv_pq):
        for j_idx, j in enumerate(pv_pq):
            if i == j:
                sum_term = 0
                for k in range(nbus):
                    if k != i:
                        Gik = Ybus[i, k].real
                        Bik = Ybus[i, k].imag
                        angle = delta[i] - delta[k]
                        sum_term += V[i] * V[k] * (Gik * np.sin(angle) - Bik * np.cos(angle))
                J1[i_idx, j_idx] = -sum_term
            else:
                Gij = Ybus[i, j].real
                Bij = Ybus[i, j].imag
                angle = delta[i] - delta[j]
                J1[i_idx, j_idx] = V[i] * V[j] * (Gij * np.sin(angle) - Bij * np.cos(angle))

    for i_idx, i in enumerate(pv_pq):
        for j_idx, j in enumerate(pq_buses):
            if i == j:
                sum_term = 0
                for k in range(nbus):
                    Gik = Ybus[i, k].real
                    Bik = Ybus[i, k].imag
                    angle = delta[i] - delta[k]
                    sum_term += V[k] * (Gik * np.cos(angle) + Bik * np.sin(angle))
                J2[i_idx, j_idx] = sum_term + 2 * V[i] * Ybus[i, i].real
            else:
                Gij = Ybus[i, j].real
                Bij = Ybus[i, j].imag
                angle = delta[i] - delta[j]
                J2[i_idx, j_idx] = V[i] * (Gij * np.cos(angle) + Bij * np.sin(angle))

    for i_idx, i in enumerate(pq_buses):
        for j_idx, j in enumerate(pv_pq):
            if i == j:
                sum_term = 0
                for k in range(nbus):
                    if k != i:
                        Gik = Ybus[i, k].real
                        Bik = Ybus[i, k].imag
                        angle = delta[i] - delta[k]
                        sum_term += V[i] * V[k] * (Gik * np.cos(angle) + Bik * np.sin(angle))
                J3[i_idx, j_idx] = sum_term
            else:
                Gij = Ybus[i, j].real
                Bij = Ybus[i, j].imag
                angle = delta[i] - delta[j]
                J3[i_idx, j_idx] = -V[i] * V[j] * (Gij * np.cos(angle) + Bij * np.sin(angle))

    for i_idx, i in enumerate(pq_buses):
        for j_idx, j in enumerate(pq_buses):
            if i == j:
                sum_term = 0
                for k in range(nbus):
                    Gik = Ybus[i, k].real
                    Bik = Ybus[i, k].imag
                    angle = delta[i] - delta[k]
                    sum_term += V[k] * (Gik * np.sin(angle) - Bik * np.cos(angle))
                J4[i_idx, j_idx] = -sum_term - 2 * V[i] * Ybus[i, i].imag
            else:
                Gij = Ybus[i, j].real
                Bij = Ybus[i, j].imag
                angle = delta[i] - delta[j]
                J4[i_idx, j_idx] = V[i] * (Gij * np.sin(angle) - Bij * np.cos(angle))

    # Jacobiana global
    J = np.block([[J1, J2], [J3, J4]])
    dx = np.linalg.solve(J, mismatch)
    # NOUS VALORS PER TENSIÓ I ANGLES
    delta[pv_pq] += dx[:n_pv_pq]
    delta[0] = 0.0
    V[pq_buses] += dx[n_pv_pq:]

# --- FIXED BRANCH FLOW FUNCTION ---
def calcular_flux_linia(i, j, V, delta, Ybus):
    '''
    Calcula el flux de potència complexa S_ij entre busos i i j.
    Retorna (P_ij, Q_ij) en p.u.
    '''
    Vi = V[i] * np.exp(1j * delta[i])
    Vj = V[j] * np.exp(1j * delta[j])
    Y_ij = -Ybus[i, j]  # Correct sign: Y_ij is positive branch admittance
    Iij = (Vi - Vj) * Y_ij
    Sij = Vi * np.conj(Iij)
    return Sij.real, Sij.imag

# -------------------------------
# Resultats per bus
print("\n--- Resultats finals per bus ---")
for i in range(nbus):
    print(f"Bus {i+1}: V = {V[i]:.4f} p.u., Angle = {np.degrees(delta[i]):.2f}°")

# -------------------------------
# Fluxos per línia
linies = [
    (0, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (4, 5),
    (4, 6),
    (6, 7),
    (7, 8)
]

print("\n--- Fluxos inversos de potència per línia ---")
for (i, j) in linies:
    Pij, Qij = calcular_flux_linia(i, j, V, delta, Ybus)
    print(f"Línia {i+1} → {j+1}: P = {Pij:.4f} p.u., Q = {Qij:.4f} p.u.")

print("\n--- Fluxos de potència per línia ---")
for (i, j) in linies:
    Pji, Qji = calcular_flux_linia(j, i, V, delta, Ybus)
    print(f"Línia {j+1} → {i+1}: P = {Pji:.4f} p.u., Q = {Qji:.4f} p.u.")

def calcular_flux_linia_Real(i, j, V, delta, Ybus):
    '''
    Calcula el flux de potència complexa S_ij entre busos i i j.
    Retorna (P_ij, Q_ij) en MW/MVAr.
    '''
    Vi = V[i] * np.exp(1j * delta[i])
    Vj = V[j] * np.exp(1j * delta[j])
    Y_ij = -Ybus[i, j]
    Iij = (Vi - Vj) * Y_ij
    Sij = Vi * np.conj(Iij)
    return 360 * Sij.real, 360 * Sij.imag

print("\n--- Fluxos de potència per línia (MW/MVAr) ---")
for (i, j) in linies:
    Pij, Qij = calcular_flux_linia_Real(i, j, V, delta, Ybus)
    print(f"Línia {i+1} → {j+1}: P = {Pij:.4f} MW, Q = {Qij:.4f} MVAr")

def pot_real_teorica(Vi, Vj, theta_i, theta_j, Gij, Bij):
    angle = theta_i - theta_j
    return Vi * Vj * (Gij * np.cos(angle) + Bij * np.sin(angle))

i = 1
j = 2
Vi = V[i]
Vj = V[j]
theta_i = delta[i]
theta_j = delta[j]
Yij = Ybus[i, j]
Gij = Yij.real
Bij = Yij.imag
Pij_teor = pot_real_teorica(Vi, Vj, theta_i, theta_j, Gij, Bij)
Pij_directe, _ = calcular_flux_linia(i, j, V, delta, Ybus)

print(f"Pij_teòrica: {Pij_teor:.4f} p.u.")
print(f"Pij_directa (Vi*Iij*): {Pij_directe:.4f} p.u.")

# --- Pèrdues de potència per línia i totals (en MW i MVAr) ---
S_base = 360  # MVA

perdua_activa_total = 0.0
perdua_reactiva_total = 0.0

print("\n--- Pèrdues per línia ---")
for (i, j) in linies:
    Pij, Qij = calcular_flux_linia(i, j, V, delta, Ybus)
    Pji, Qji = calcular_flux_linia(j, i, V, delta, Ybus)

    P_perdua_pu = Pij + Pji
    Q_perdua_pu = Qij + Qji

    P_perdua_MW = P_perdua_pu * S_base
    Q_perdua_MVAr = Q_perdua_pu * S_base

    print(f"Línia {i+1} ↔ {j+1}: Pèrdua = {P_perdua_MW:.2f} MW, Q = {Q_perdua_MVAr:.2f} MVAr")

    perdua_activa_total += P_perdua_pu
    perdua_reactiva_total += Q_perdua_pu

# Totals
P_total_MW = perdua_activa_total * S_base
Q_total_MVAr = perdua_reactiva_total * S_base

print("\n⚡️ PÈRDUES TOTALS AL SISTEMA:")
print(f"Potència activa perduda:   {P_total_MW:.2f} MW")
print(f"Potència reactiva perduda: {Q_total_MVAr:.2f} MVAr")
