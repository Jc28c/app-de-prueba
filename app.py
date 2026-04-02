# =============================================================================
# PROYECTO: OPTIMIZACIÓN Y SIMULACIÓN DIGITAL DE SISTEMAS DE TRANSPORTE DE GAS
# =============================================================================
# Universidad Central de Venezuela - Escuela de Ingeniería Química
# Optimización de Procesos - Semestre 3-2025
# Prof. Ricardo Olejnik
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Simulador Gasoducto Trans-Andino",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# ESTILO PERSONALIZADO (fondo claro, texto oscuro, acentos azul-morado)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Fondo general claro */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
    }
    /* Texto principal oscuro */
    html, body, [class*="css"] {
        color: #1e293b;
    }
    /* Sidebar con fondo semitransparente claro */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(8px);
        border-right: 1px solid #cbd5e1;
    }
    /* Títulos con gradiente azul-morado */
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -0.02em;
        color: #0f172a;
    }
    h1 {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    /* Métricas: valor con degradado y texto oscuro */
    div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    div[data-testid="stMetricLabel"] {
        color: #334155 !important;
    }
    div[data-testid="stMetricDelta"] {
        color: #475569 !important;
    }
    /* Botones */
    .stButton button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        border: none;
        color: white;
        font-weight: 500;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
    }
    /* Alertas con bordes redondeados */
    .stAlert {
        border-radius: 12px;
        border-left: 5px solid;
    }
    /* Dataframe y tablas */
    .stDataFrame {
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    /* Captions y texto auxiliar */
    .small, .stCaption {
        color: #475569 !important;
    }
    /* Enlaces */
    a {
        color: #4f46e5;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DATOS TÉCNICOS FIJOS (Tablas del enunciado)
# -----------------------------------------------------------------------------
TUBERIAS = pd.DataFrame({
    'DN_pulg': [12, 16, 20, 24],
    'DE_mm': [323.8, 406.4, 508.0, 609.6],
    'espesor_mm': [10.31, 12.70, 15.09, 17.48],
    'costo_USD_m': [185, 260, 350, 440]
})
# Conversión a pulgadas para cálculos
TUBERIAS['DE_pulg'] = TUBERIAS['DE_mm'] / 25.4
TUBERIAS['espesor_pulg'] = TUBERIAS['espesor_mm'] / 25.4

GRADOS = pd.DataFrame({
    'grado': ['X52', 'X60'],
    'SMYS_psi': [52000, 60000],
    'F': [0.72, 0.72]
})

# Constantes físicas y económicas base
L_km = 400.0                # km
L_millas = L_km * 0.621371  # millas
Q_base = 500.0              # MMscfd (diseño, pero puede variar)
P_in = 800.0                # psia
P_out_min = 500.0           # psia
T_suction_K = 293.15        # K
T_suction_R = T_suction_K * 9/5  # Rankine = 527.67
gamma = 0.65
Z = 0.90
E_tuberia = 0.95            # eficiencia Weymouth
k = 1.25                    # relación cp/cv para gas natural
eta_comp = 0.85             # eficiencia compresor
R_gas = 10.7316             # psia·ft³/(lbmol·°R) (constante universal)
# Factor de conversión para potencia: 1 HP = 0.7457 kW
HP_to_kW = 0.7457
horas_anuales = 8760        # h/año
vida_util = 20              # años

# -----------------------------------------------------------------------------
# FUNCIONES AUXILIARES
# -----------------------------------------------------------------------------
def calcular_maop(smys_psi, espesor_pulg, de_pulg, F):
    """Límite de presión de operación (Barlow) en psia."""
    # MAOP = 2 * SMYS * t * F / D (en unidades consistentes)
    return 2 * smys_psi * espesor_pulg * F / de_pulg

def calcular_caida_presion(Q, L_millas, D_pulg, P_entrada):
    """
    Usa la ecuación de Weymouth para calcular la presión de salida.
    Devuelve P_salida en psia.
    """
    # Weymouth: P1² - P2² = 433.5 * (Q/E)² * (L * γ * T * Z) / D^5.33
    term = 433.5 * (Q / E_tuberia)**2
    term *= (L_millas * gamma * T_suction_R * Z) / (D_pulg**5.33)
    P2_sq = P_entrada**2 - term
    if P2_sq <= 0:
        return 0.0
    return np.sqrt(P2_sq)

def calcular_perfil_presion(Q, D_pulg, L_km_total, N_estaciones, P_inicial, P_out_min):
    """
    Calcula el perfil de presión a lo largo de la tubería.
    Parámetros:
        Q: flujo (MMscfd)
        D_pulg: diámetro interno (pulgadas)
        L_km_total: longitud total (km)
        N_estaciones: número de estaciones de compresión (incluye la inicial)
        P_inicial: presión de recepción (psia)
        P_out_min: presión mínima requerida al final (psia)
    Retorna:
        distancias_km: array de distancias (km)
        presiones_psia: array de presiones (psia)
        r_comp: relación de compresión por estación (asumida uniforme)
        P_descarga: presión de descarga de cada estación (psia)
    """
    if N_estaciones < 1:
        N_estaciones = 1
    # Longitud por tramo (km)
    L_tramo_km = L_km_total / N_estaciones
    L_tramo_millas = L_tramo_km * 0.621371

    # Relación de compresión necesaria para cumplir P_out_min
    r_min = 1.0
    r_max = 3.0
    for _ in range(50):
        r = (r_min + r_max) / 2
        P_actual = P_inicial
        for i in range(N_estaciones):
            # Compresión
            P_desc = P_actual * r
            # Caída en el tramo
            P_salida = calcular_caida_presion(Q, L_tramo_millas, D_pulg, P_desc)
            if P_salida <= 0:
                P_salida = 0.0
                break
            P_actual = P_salida
        if P_actual >= P_out_min:
            r_max = r
        else:
            r_min = r
        if r_max - r_min < 1e-6:
            break
    r_comp = (r_min + r_max) / 2

    # Construir perfil completo
    distancias = []
    presiones = []
    P_actual = P_inicial
    dist_km = 0.0
    for est in range(N_estaciones):
        # Punto de succión (antes de comprimir)
        distancias.append(dist_km)
        presiones.append(P_actual)
        # Compresión
        P_desc = P_actual * r_comp
        distancias.append(dist_km)  # mismo punto, presión salto
        presiones.append(P_desc)
        # Avanzar tramo
        dist_km += L_tramo_km
        # Caída en el tramo
        P_salida = calcular_caida_presion(Q, L_tramo_millas, D_pulg, P_desc)
        if P_salida <= 0:
            P_salida = 0.0
        P_actual = P_salida
    # Añadir punto final
    distancias.append(dist_km)
    presiones.append(P_actual)

    return np.array(distancias), np.array(presiones), r_comp, P_desc

def calcular_potencia_estacion(Q, r, T_suction_R, Z, eta):
    """Potencia de una estación de compresión (HP) usando fórmula simplificada."""
    term = r**((k-1)/k) - 1
    HP = (0.0857 * Q * Z * T_suction_R) / (eta * (k-1)) * term
    return max(0, HP)

def calcular_capex(DN_pulg, costo_tuberia_USD_m, N_estaciones):
    capex_tuberia = L_km * 1000 * costo_tuberia_USD_m
    return capex_tuberia

def calcular_opex(potencia_total_HP, costo_energia_USD_kWh):
    potencia_kW = potencia_total_HP * HP_to_kW
    consumo_anual_kWh = potencia_kW * horas_anuales
    opex_energia = consumo_anual_kWh * costo_energia_USD_kWh
    return opex_energia

def calcular_tac(capex, opex, tasa_anual, vida_anos):
    if tasa_anual == 0:
        crf = 1 / vida_anos
    else:
        crf = tasa_anual * (1 + tasa_anual)**vida_anos / ((1 + tasa_anual)**vida_anos - 1)
    tac = capex * crf + opex
    return tac

# -----------------------------------------------------------------------------
# SIDEBAR: CONFIGURACIÓN DEL USUARIO
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuración del Gasoducto")
    st.markdown("---")

    st.image("logo_gd.jpeg", width=100)
    st.markdown('<div class="logo_gd-sidebar"><span>GD</span><br<small>El Gemelo Digital</small></div>', unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("📊 Parámetros Económicos")
    costo_energia = st.number_input("Costo de energía (USD/kWh)", min_value=0.01, max_value=1.0, value=0.08, step=0.01, format="%.3f")
    factor_costo_acero = st.slider("Factor costo acero", min_value=0.5, max_value=2.0, value=1.0, step=0.05, help="Multiplicador del costo base del tubo")
    tasa_interes = st.number_input("Tasa de interés anual (%)", min_value=0.0, max_value=30.0, value=8.0, step=0.5, format="%.1f") / 100.0

    st.markdown("---")
    st.subheader("🔧 Selección de Material")
    dn_opcion = st.selectbox("Diámetro nominal (pulg)", options=TUBERIAS['DN_pulg'].tolist(), index=2, format_func=lambda x: f"{x}\"")
    grado_opcion = st.selectbox("Grado de acero", options=GRADOS['grado'].tolist(), index=0)

    tuberia_sel = TUBERIAS[TUBERIAS['DN_pulg'] == dn_opcion].iloc[0]
    DE_pulg = tuberia_sel['DE_pulg']
    espesor_pulg = tuberia_sel['espesor_pulg']
    costo_base_tubo = tuberia_sel['costo_USD_m'] * factor_costo_acero
    st.caption(f"Costo tubo: ${costo_base_tubo:.2f}/m (ajustado)")

    grado_sel = GRADOS[GRADOS['grado'] == grado_opcion].iloc[0]
    SMYS = grado_sel['SMYS_psi']
    F = grado_sel['F']

    st.markdown("---")
    st.subheader("🔄 Variables Operativas")
    Q_oper = st.number_input("Flujo de gas (MMscfd)", min_value=100.0, max_value=1500.0, value=Q_base, step=10.0)
    N_estaciones = st.slider("Número de estaciones de compresión", min_value=1, max_value=10, value=2, step=1, help="Incluye la estación inicial")

    st.markdown("---")
    st.markdown("**Resumen del diseño**")
    st.write(f"Longitud: **{L_km} km**")
    st.write(f"Presión recepción: **{P_in} psia**")
    st.write(f"Presión mínima entrega: **{P_out_min} psia**")
    st.write(f"Gas: γ = {gamma}, Z = {Z}, T₁ = {T_suction_K} K")

# -----------------------------------------------------------------------------
# CÁLCULOS PRINCIPALES
# -----------------------------------------------------------------------------
DI_pulg = DE_pulg - 2 * espesor_pulg
MAOP = calcular_maop(SMYS, espesor_pulg, DE_pulg, F)

distancias_km, presiones_psia, r_comp, _ = calcular_perfil_presion(
    Q=Q_oper, D_pulg=DI_pulg, L_km_total=L_km,
    N_estaciones=N_estaciones, P_inicial=P_in, P_out_min=P_out_min
)

P_succ = []
for i in range(len(distancias_km)):
    if i % 2 == 0:
        P_succ.append(presiones_psia[i])
if len(P_succ) < N_estaciones:
    P_succ.extend([P_succ[-1]] * (N_estaciones - len(P_succ)))
else:
    P_succ = P_succ[:N_estaciones]

potencia_por_estacion_HP = [calcular_potencia_estacion(Q_oper, r_comp, T_suction_R, Z, eta_comp) for _ in range(N_estaciones)]
potencia_total_HP = sum(potencia_por_estacion_HP)

T_desc_R = T_suction_R * (r_comp**((k-1)/k))
T_desc_C = (T_desc_R - 459.67) * 5/9

capex_tuberia = L_km * 1000 * costo_base_tubo
costo_compresor_por_HP = 2000.0
capex_compresores = potencia_total_HP * costo_compresor_por_HP
capex_total = capex_tuberia + capex_compresores

opex_energia = calcular_opex(potencia_total_HP, costo_energia)
tac = calcular_tac(capex_total, opex_energia, tasa_interes, vida_util)

# -----------------------------------------------------------------------------
# PANEL PRINCIPAL
# -----------------------------------------------------------------------------
st.title("🛢️ Simulador del Gasoducto Trans‑Andino")
st.markdown("""
Aplicación interactiva para la optimización del transporte de gas natural utilizando el método de Weymouth.
Ajusta los parámetros en la barra lateral y observa los efectos en tiempo real.
""")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Costo Total Anualizado (TAC)", f"${tac:,.0f}")
with col2:
    st.metric("Potencia Total Instalada", f"{potencia_total_HP:,.0f} HP")
with col3:
    P_final = presiones_psia[-1]
    st.metric("Presión de Entrega", f"{P_final:.1f} psia", delta=f"{P_final - P_out_min:.1f}")
with col4:
    st.metric("Relación de Compresión", f"{r_comp:.3f}")

# --- Gráfico de perfil hidráulico (Plotly con plantilla clara) ---
st.markdown("## 📈 Perfil de Presión a lo largo del Gasoducto")
fig_pres = go.Figure()
fig_pres.add_trace(go.Scatter(
    x=distancias_km, y=presiones_psia,
    mode='lines+markers',
    name='Presión (psia)',
    line=dict(color='#4f46e5', width=3),
    marker=dict(size=6, color='#7c3aed'),
    hovertemplate='Distancia: %{x:.1f} km<br>Presión: %{y:.1f} psia<extra></extra>'
))
fig_pres.add_hline(y=P_out_min, line_dash="dash", line_color="red", annotation_text="Presión mínima de entrega")
fig_pres.add_hline(y=MAOP, line_dash="dash", line_color="orange", annotation_text=f"MAOP = {MAOP:.0f} psia")

fig_pres.update_layout(
    title="Evolución de la presión con estaciones de compresión",
    xaxis_title="Distancia (km)",
    yaxis_title="Presión (psia)",
    template="plotly_white",
    hovermode="x unified",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#1e293b')
)
st.plotly_chart(fig_pres, use_container_width=True)

# --- Desglose de costos ---
st.markdown("## 💰 Desglose de Costos")
capex_componentes = {'Tubería': capex_tuberia, 'Compresores': capex_compresores}
opex_componentes = {'Energía': opex_energia}
df_capex = pd.DataFrame(list(capex_componentes.items()), columns=['Concepto', 'CAPEX (USD)'])
df_opex = pd.DataFrame(list(opex_componentes.items()), columns=['Concepto', 'OPEX anual (USD)'])

col_bar1, col_bar2 = st.columns(2)
with col_bar1:
    fig_capex = px.bar(df_capex, x='Concepto', y='CAPEX (USD)', color='Concepto',
                       color_discrete_sequence=['#4f46e5', '#7c3aed'],
                       title='Inversión de Capital (CAPEX)')
    fig_capex.update_layout(template='plotly_white', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#1e293b'))
    st.plotly_chart(fig_capex, use_container_width=True)
with col_bar2:
    fig_opex = px.bar(df_opex, x='Concepto', y='OPEX anual (USD)', color='Concepto',
                      color_discrete_sequence=['#4f46e5'],
                      title='Costos Operativos Anuales (OPEX)')
    fig_opex.update_layout(template='plotly_white', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#1e293b'))
    st.plotly_chart(fig_opex, use_container_width=True)

# --- Alertas de validación ---
st.markdown("## ⚠️ Validación del Diseño")
col_alert1, col_alert2, col_alert3 = st.columns(3)
presion_max = np.max(presiones_psia)
if presion_max > MAOP:
    col_alert1.error(f"❌ **MAOP excedido**\nPresión máxima: {presion_max:.0f} psia > MAOP = {MAOP:.0f} psia")
else:
    col_alert1.success(f"✅ **MAOP OK**\nPresión máxima: {presion_max:.0f} psia ≤ {MAOP:.0f} psia")

if T_desc_C > 65:
    col_alert2.error(f"❌ **Temperatura excesiva**\nT₂ = {T_desc_C:.1f} °C > 65 °C")
else:
    col_alert2.success(f"✅ **Temperatura OK**\nT₂ = {T_desc_C:.1f} °C ≤ 65 °C")

if P_final < P_out_min:
    col_alert3.error(f"❌ **Presión de entrega insuficiente**\n{P_final:.1f} psia < {P_out_min} psia")
else:
    col_alert3.success(f"✅ **Presión de entrega OK**\n{P_final:.1f} psia ≥ {P_out_min} psia")

# --- Tabla de resultados técnicos ---
st.markdown("## 📊 Resultados Técnicos Detallados")
datos_tecnicos = {
    'Parámetro': [
        'Flujo (MMscfd)', 'Diámetro interno (pulg)', 'Espesor (pulg)',
        'Grado de acero', 'SMYS (psi)', 'MAOP (psia)',
        'Relación de compresión', 'N° estaciones',
        'Potencia total (HP)', 'Temperatura descarga (°C)',
        'Presión final (psia)', 'CAPEX total (USD)', 'OPEX anual (USD)', 'TAC (USD/año)'
    ],
    'Valor': [
        f"{Q_oper:.1f}", f"{DI_pulg:.2f}", f"{espesor_pulg:.3f}",
        grado_opcion, f"{SMYS}", f"{MAOP:.0f}",
        f"{r_comp:.3f}", f"{N_estaciones}",
        f"{potencia_total_HP:.0f}", f"{T_desc_C:.1f}",
        f"{P_final:.1f}", f"${capex_total:,.0f}", f"${opex_energia:,.0f}", f"${tac:,.0f}"
    ]
}
df_tec = pd.DataFrame(datos_tecnicos)
st.dataframe(df_tec, use_container_width=True, hide_index=True)

# --- Exportación CSV ---
st.markdown("---")
st.subheader("📥 Exportar resultados")
export_data = {
    'Distancia_km': distancias_km,
    'Presion_psia': presiones_psia
}
df_export = pd.DataFrame(export_data)
df_export_params = pd.DataFrame({
    'Parametro': ['Q_MMscfd', 'DN_pulg', 'Grado', 'N_estaciones', 'r_comp', 'Potencia_total_HP', 'TAC_USD'],
    'Valor': [Q_oper, dn_opcion, grado_opcion, N_estaciones, r_comp, potencia_total_HP, tac]
})
csv_buffer = StringIO()
df_export.to_csv(csv_buffer, index=False)
csv_buffer.write('\n')
df_export_params.to_csv(csv_buffer, index=False)
st.download_button(
    label="📄 Descargar resultados completos (CSV)",
    data=csv_buffer.getvalue(),
    file_name="simulacion_gasoducto.csv",
    mime="text/csv"
)

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 15px; color: #4b5563; font-size: 0.85rem;">
    <strong>Universidad Central de Venezuela - Escuela de Ingeniería Química</strong><br>
    Optimización de Procesos • Prof. Ricardo Olejnik • 2026
</div>
""", unsafe_allow_html=True)
