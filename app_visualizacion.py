"""
App de visualización: Dashboard de gastos médicos.
Curso de Ciencia de Datos - Ingeniería Industrial.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

# ---------- Configuración general ----------
st.set_page_config(
    page_title="Gastos médicos",
    page_icon="🏥",
    layout="wide"
)

# ---------- Configuración reutilizable ----------
COLOR_FUMADOR = {
    "si": "#ea580c",
    "no": "#1e3a8a"
}

# ---------- Carga de datos con caché ----------
@st.cache_data
def cargar_datos(ruta: str) -> pd.DataFrame:
    """Carga el CSV una sola vez por sesión."""
    return pd.read_csv(ruta)

df = cargar_datos("data/gastos_medicos.csv")

# ---------- Encabezado ----------
st.title("🏥 Dashboard de gastos médicos")
st.caption("Análisis exploratorio sobre 1338 pacientes asegurados.")

# ---------- Sidebar: filtros ----------
with st.sidebar:

    st.header("🎛️ Filtros")

    sexo_sel = st.multiselect(
        "Sexo",
        df["sexo"].unique(),
        default=list(df["sexo"].unique())
    )

    fumador_sel = st.multiselect(
        "¿Es fumador?",
        df["fumador"].unique(),
        default=list(df["fumador"].unique())
    )

    region_sel = st.multiselect(
        "Región",
        df["region"].unique(),
        default=list(df["region"].unique())
    )

    edad_min, edad_max = st.slider(
        "Rango de edad",
        int(df["edad"].min()),
        int(df["edad"].max()),
        (
            int(df["edad"].min()),
            int(df["edad"].max())
        )
    )

    hijos_sel = st.slider(
        "Número mínimo de hijos",
        int(df["hijos"].min()),
        int(df["hijos"].max()),
        int(df["hijos"].min())
    )

# ---------- Aplicar filtros ----------
mask = (
    df["sexo"].isin(sexo_sel)
    & df["fumador"].isin(fumador_sel)
    & df["region"].isin(region_sel)
    & df["edad"].between(edad_min, edad_max)
    & (df["hijos"] >= hijos_sel)
)

dff = df[mask]

# ---------- Validación ----------
if len(dff) == 0:
    st.warning(
        "⚠️ No hay registros con esos filtros. Ajusta los criterios."
    )
    st.stop()

# ---------- Variables auxiliares ----------
N = len(dff)

porcentaje_fumadores = (
    (dff["fumador"] == "si").mean() * 100
)

# ---------- Mensaje contextual ----------
if porcentaje_fumadores > 30:
    st.info(
        "Esta población tiene una proporción alta de fumadores."
    )

# ---------- KPIs ----------
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Pacientes filtrados",
    f"{N:,}"
)

c2.metric(
    "Gasto promedio",
    f"${dff['gastos'].mean():,.0f}"
)

c3.metric(
    "Gasto mediano",
    f"${dff['gastos'].median():,.0f}"
)

c4.metric(
    "% fumadores",
    f"{porcentaje_fumadores:.1f}%"
)

c5.metric(
    "Edad promedio",
    f"{dff['edad'].mean():.1f} años"
)

st.divider()

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs([
    "📊 Distribuciones",
    "🔗 Relaciones",
    "📋 Datos"
])

# =========================================================
# TAB 1
# =========================================================
with tab1:

    col1, col2 = st.columns(2)

    # ---------- Histograma ----------
    with col1:

        fig_hist = px.histogram(
            dff,
            x="gastos",
            nbins=40,
            title=f"Distribución de gastos médicos (n={N})",
            color_discrete_sequence=["#7c3aed"]
        )

        st.plotly_chart(
            fig_hist,
            width="stretch"
        )

    # ---------- Boxplot fumadores ----------
    with col2:

        fig_box = px.box(
            dff,
            x="fumador",
            y="gastos",
            color="fumador",
            title=f"Gastos según hábito de fumar (n={N})",
            color_discrete_map=COLOR_FUMADOR
        )

        st.plotly_chart(
            fig_box,
            width="stretch"
        )

    # ---------- Boxplot por región ----------
    fig_region = px.box(
        dff,
        x="region",
        y="gastos",
        color="region",
        title=f"Gastos por región (n={N})"
    )

    st.plotly_chart(
        fig_region,
        width="stretch"
    )

    # ---------- Barras por región ----------
    region_counts = (
        dff["region"]
        .value_counts()
        .reset_index()
    )

    region_counts.columns = [
        "region",
        "cantidad"
    ]

    fig_barras = px.bar(
        region_counts,
        x="region",
        y="cantidad",
        color="region",
        title=f"Cantidad de pacientes por región (n={N})"
    )

    st.plotly_chart(
        fig_barras,
        width="stretch"
    )

# =========================================================
# TAB 2
# =========================================================
with tab2:

    # ---------- Edad vs gastos ----------
    fig_scatter = px.scatter(
        dff,
        x="edad",
        y="gastos",
        color="fumador",
        size="imc",
        hover_data=[
            "sexo",
            "region",
            "hijos"
        ],
        title=f"Edad vs. gastos (n={N})",
        color_discrete_map=COLOR_FUMADOR
    )

    st.plotly_chart(
        fig_scatter,
        width="stretch"
    )

    # ---------- IMC vs gastos ----------
    fig_imc = px.scatter(
        dff,
        x="imc",
        y="gastos",
        color="fumador",
        title=f"IMC vs. gastos (n={N})",
        color_discrete_map=COLOR_FUMADOR,
        trendline="ols"
    )

    st.plotly_chart(
        fig_imc,
        width="stretch"
    )

    # ---------- Violín ----------
    fig_violin = px.violin(
        dff,
        x="sexo",
        y="gastos",
        color="fumador",
        box=True,
        points="all",
        title=f"Gastos por sexo y hábito de fumar (n={N})",
        color_discrete_map=COLOR_FUMADOR
    )

    st.plotly_chart(
        fig_violin,
        width="stretch"
    )

# =========================================================
# TAB 3
# =========================================================
with tab3:

    st.dataframe(
        dff,
        width="stretch",
        height=400
    )

    csv = dff.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Descargar datos filtrados (CSV)",
        csv,
        "gastos_filtrados.csv",
        "text/csv"
    )