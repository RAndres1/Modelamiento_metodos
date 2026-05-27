import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# =====================================================
# 1. PARAMETROS DEL MODELO
# =====================================================

r = 0.8
K = 100
alpha = 0.005
beta = 0.8
m = 0.2
gamma = 0.02
delta = 0.2
n = 0.1
lambda_P = 1
lambda_C = 1
lambda_Z = 1

# =====================================================
# 2. SISTEMA DE ECUACIONES
# =====================================================
# Y = [P, u, C, v, Z, w]
# P = pasto
# u = velocidad de cambio del pasto
# C = conejos
# v = velocidad de cambio de conejos
# Z = zorros
# w = velocidad de cambio de zorros

def sistema(t, Y):
    P, u, C, v, Z, w = Y
    dP = u
    du = r * P * (1 - P / K) - alpha * P * C - lambda_P * u
    dC = v
    dv = beta * alpha * P * C - m * C - gamma * C * Z - lambda_C * v
    dZ = w
    dw = delta * gamma * C * Z - n * Z - lambda_Z * w
    return np.array([dP, du, dC, dv, dZ, dw])

# =====================================================
# 3. METODO DE EULER
# =====================================================

def euler(f, t0, tf, Y0, h):
    t = np.arange(t0, tf + h, h)
    Y = np.zeros((len(t), len(Y0)))
    Y[0] = Y0
    for i in range(len(t) - 1):
        Y[i + 1] = Y[i] + h * f(t[i], Y[i])
    return t, Y

# =====================================================
# 4. EULER MEJORADO
# =====================================================

def euler_mejorado(f, t0, tf, Y0, h):
    t = np.arange(t0, tf + h, h)
    Y = np.zeros((len(t), len(Y0)))
    Y[0] = Y0
    for i in range(len(t) - 1):
        k1 = f(t[i], Y[i])
        prediccion = Y[i] + h * k1
        k2 = f(t[i] + h, prediccion)
        Y[i + 1] = Y[i] + (h / 2) * (k1 + k2)
    return t, Y

# =====================================================
# 5. METODO RK4
# =====================================================

def rk4(f, t0, tf, Y0, h):
    t = np.arange(t0, tf + h, h)
    Y = np.zeros((len(t), len(Y0)))
    Y[0] = Y0
    for i in range(len(t) - 1):
        k1 = f(t[i], Y[i])
        k2 = f(t[i] + h / 2, Y[i] + h * k1 / 2)
        k3 = f(t[i] + h / 2, Y[i] + h * k2 / 2)
        k4 = f(t[i] + h, Y[i] + h * k3)
        Y[i + 1] = Y[i] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return t, Y

# =====================================================
# 6. DATOS INICIALES DEL CASO BASE
# =====================================================

t0 = 0
tf = 100
h = 0.1
Y0 = np.array([80, 0, 20, 0, 5, 0])

# =====================================================
# 7. EJECUTAR LOS TRES METODOS
# =====================================================

t, Y_euler = euler(sistema, t0, tf, Y0, h)
t, Y_em = euler_mejorado(sistema, t0, tf, Y0, h)
t, Y_rk4 = rk4(sistema, t0, tf, Y0, h)

# =====================================================
# 8. SOLUCION DE REFERENCIA CON RK45
# =====================================================

ref = solve_ivp(
    sistema,
    (t0, tf),
    Y0,
    method="RK45",
    t_eval=t
)
Y_ref = ref.y.T


# =====================================================
# 9. ERROR EN LA POBLACION DE CONEJOS
# =====================================================

error_euler = abs(Y_euler[:, 2] - Y_ref[:, 2])
error_em = abs(Y_em[:, 2] - Y_ref[:, 2])
error_rk4 = abs(Y_rk4[:, 2] - Y_ref[:, 2])

print("ERROR DEL CASO BASE EN LA POBLACION DE CONEJOS")
print("Comparación contra RK45")
print("Paso h =", h)
print("----------------------------------")
print("Euler - Error promedio:", np.mean(error_euler))
print("Euler - Error máximo:", np.max(error_euler))
print("----------------------------------")
print("Euler mejorado - Error promedio:", np.mean(error_em))
print("Euler mejorado - Error máximo:", np.max(error_em))
print("----------------------------------")
print("RK4 - Error promedio:", np.mean(error_rk4))
print("RK4 - Error máximo:", np.max(error_rk4))
print("----------------------------------")


# =====================================================
# 10. GRAFICA DEL MODELO CON RK4
# =====================================================

plt.figure(figsize=(10, 6))
plt.plot(t, Y_rk4[:, 0], label="Pasto")
plt.plot(t, Y_rk4[:, 2], label="Conejos")
plt.plot(t, Y_rk4[:, 4], label="Zorros")
plt.xlabel("Tiempo")
plt.ylabel("Población / biomasa")
plt.title("Modelo presa-depredador de tres especies con inercia poblacional")
plt.legend()
plt.grid()
plt.show()


# =====================================================
# 11. COMPARACION ENTRE METODOS
# =====================================================

plt.figure(figsize=(10, 6))
plt.plot(t, Y_euler[:, 2], label="Euler")
plt.plot(t, Y_em[:, 2], label="Euler mejorado")
plt.plot(t, Y_rk4[:, 2], label="RK4")
plt.plot(t, Y_ref[:, 2], "--", label="RK45 referencia")
plt.xlabel("Tiempo")
plt.ylabel("Población de conejos")
plt.title("Comparación de métodos para la población de conejos")
plt.legend()
plt.grid()
plt.show()


# =====================================================
# 12. COMPARACION DE ERRORES CON VARIOS PASOS h
# =====================================================
# Aquí se repite el cálculo con varios tamaños de paso.
# La idea es ver que, cuando h disminuye, el error también baja.

pasos = [0.5, 0.2, 0.1, 0.05]

print("\nCOMPARACION DE ERRORES CON DIFERENTES PASOS")
print("Error promedio en la población de conejos")
print("------------------------------------------------------------")
print("h\tEuler\t\tEuler mejorado\t\tRK4")
print("------------------------------------------------------------")

errores_euler = []
errores_em = []
errores_rk4 = []

for h_actual in pasos:
    t_actual, Y_euler_actual = euler(sistema, t0, tf, Y0, h_actual)
    t_actual, Y_em_actual = euler_mejorado(sistema, t0, tf, Y0, h_actual)
    t_actual, Y_rk4_actual = rk4(sistema, t0, tf, Y0, h_actual)

    ref_actual = solve_ivp(
        sistema,
        (t0, tf),
        Y0,
        method="RK45",
        t_eval=t_actual
    )

    Y_ref_actual = ref_actual.y.T

    error_euler_actual = np.mean(abs(Y_euler_actual[:, 2] - Y_ref_actual[:, 2]))
    error_em_actual = np.mean(abs(Y_em_actual[:, 2] - Y_ref_actual[:, 2]))
    error_rk4_actual = np.mean(abs(Y_rk4_actual[:, 2] - Y_ref_actual[:, 2]))

    errores_euler.append(error_euler_actual)
    errores_em.append(error_em_actual)
    errores_rk4.append(error_rk4_actual)

    print(h_actual, "\t", round(error_euler_actual, 6), "\t", round(error_em_actual, 6), "\t\t", round(error_rk4_actual, 6))


# Grafica de error contra tamaño de paso

plt.figure(figsize=(10, 6))
plt.plot(pasos, errores_euler, marker="o", label="Euler")
plt.plot(pasos, errores_em, marker="o", label="Euler mejorado")
plt.plot(pasos, errores_rk4, marker="o", label="RK4")
plt.xlabel("Tamaño de paso h")
plt.ylabel("Error promedio en conejos")
plt.title("Comparación del error según el tamaño de paso")
plt.legend()
plt.grid()
plt.gca().invert_xaxis()
plt.show()


# =====================================================
# 13. CINCO ESCENARIOS ADICIONALES
# =====================================================
# Para no complicar el análisis, usamos RK4 en los escenarios.
# RK4 es el método propio más preciso de los tres.

escenarios = {
    "Caso base": np.array([80, 0, 20, 0, 5, 0]),
    "Muchos zorros": np.array([80, 0, 20, 0, 20, 0]),
    "Pocos zorros": np.array([80, 0, 20, 0, 1, 0]),
    "Poco pasto y muchos conejos": np.array([30, 0, 60, 0, 5, 0]),
    "Mucho pasto y pocos conejos": np.array([100, 0, 5, 0, 5, 0]),
    "Pocos recursos y pocos animales": np.array([40, 0, 10, 0, 2, 0])
}

for nombre, condicion_inicial in escenarios.items():
    t_esc, Y_esc = rk4(sistema, t0, tf, condicion_inicial, h)

    plt.figure(figsize=(10, 6))
    plt.plot(t_esc, Y_esc[:, 0], label="Pasto")
    plt.plot(t_esc, Y_esc[:, 2], label="Conejos")
    plt.plot(t_esc, Y_esc[:, 4], label="Zorros")
    plt.xlabel("Tiempo")
    plt.ylabel("Población / biomasa")
    plt.title(nombre)
    plt.legend()
    plt.grid()
    plt.show()


# =====================================================
# 14. ANIMACION SENCILLA DEL CASO BASE
# =====================================================
# Esta animación muestra cómo se van formando las curvas
# de pasto, conejos y zorros con el paso del tiempo.

fig, ax = plt.subplots(figsize=(10, 6))

linea_pasto, = ax.plot([], [], label="Pasto")
linea_conejos, = ax.plot([], [], label="Conejos")
linea_zorros, = ax.plot([], [], label="Zorros")

ax.set_xlim(t0, tf)
ax.set_ylim(0, 110)
ax.set_xlabel("Tiempo")
ax.set_ylabel("Población / biomasa")
ax.set_title("Animación del modelo presa-depredador")
ax.legend()
ax.grid()


def iniciar_animacion():
    linea_pasto.set_data([], [])
    linea_conejos.set_data([], [])
    linea_zorros.set_data([], [])
    return linea_pasto, linea_conejos, linea_zorros


def actualizar_animacion(i):
    linea_pasto.set_data(t[:i], Y_rk4[:i, 0])
    linea_conejos.set_data(t[:i], Y_rk4[:i, 2])
    linea_zorros.set_data(t[:i], Y_rk4[:i, 4])
    return linea_pasto, linea_conejos, linea_zorros


animacion = FuncAnimation(
    fig,
    actualizar_animacion,
    frames=len(t),
    init_func=iniciar_animacion,
    interval=20,
    blit=True
)
animacion.save("animacion_presa_depredador.gif", writer="pillow")
plt.show()

