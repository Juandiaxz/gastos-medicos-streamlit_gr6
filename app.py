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
        "Sexo", df["sexo"].unique(),
        default=list(df["sexo"].unique())
    )
    fumador_sel = st.multiselect(
        "¿Es fumador?", df["fumador"].unique(),
        default=list(df["fumador"].unique())
    )
    region_sel = st.multiselect(
        "Región", df["region"].unique(),
        default=list(df["region"].unique())
    )
    edad_min, edad_max = st.slider(
        "Rango de edad",
        int(df["edad"].min()), int(df["edad"].max()),
        (int(df["edad"].min()), int(df["edad"].max()))
    )
    hijos_sel = st.slider(
        "Número de hijos (mín)",
        int(df["hijos"].min()), int(df["hijos"].max()),
        int(df["hijos"].min())
    )

# ---------- Aplicar filtros ----------
mask = (
    df["sexo"].isin(sexo_sel) &
    df["fumador"].isin(fumador_sel) &
    df["region"].isin(region_sel) &
    df["edad"].between(edad_min, edad_max) &
    (df["hijos"] >= hijos_sel)
)
dff = df[mask]

# Validación: si el filtro deja datos vacíos
if len(dff) == 0:
    st.warning("⚠️ No hay registros con esos filtros. Ajusta los criterios.")
    st.stop()

# ---------- KPIs ----------
# Modificado a 5 columnas para incluir edad promedio
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Pacientes filtrados", f"{len(dff):,}")
c2.metric("Gasto promedio", f"${dff['gastos'].mean():,.0f}")
c3.metric("Gasto mediano", f"${dff['gastos'].median():,.0f}")
c4.metric("Edad promedio", f"{dff['edad'].mean():.1f}")
c5.metric("% fumadores", f"{(dff['fumador'] == 'si').mean() * 100:.1f}%")

# Mensaje contextual si el % de fumadores es mayor al 30%
if (dff['fumador'] == 'si').mean() * 100 > 30:
    st.info("Esta población tiene una proporción alta de fumadores")

st.divider()

# ---------- Visualizaciones en tabs ----------
tab1, tab2, tab3 = st.tabs([
    "📊 Distribuciones",
    "🔗 Relaciones",
    "📋 Datos"
])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        # Título personalizado con n y color de preferencia (#008080 - Teal)
        fig = px.histogram(
            dff, x="gastos", nbins=40,
            title=f"Distribución de gastos médicos (n={len(dff)})",
            color_discrete_sequence=["#008080"]
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        # Título personalizado con n
        fig = px.box(
            dff, x="fumador", y="gastos", color="fumador",
            title=f"Gastos según hábito de fumar (n={len(dff)})",
            color_discrete_map={"si": "#ea580c", "no": "#1e3a8a"}
        )
        st.plotly_chart(fig, use_container_width=True)

    # Título personalizado con n
    fig_region = px.box(
        dff, x="region", y="gastos", color="region",
        title=f"Gastos por región (n={len(dff)})"
    )
    st.plotly_chart(fig_region, use_container_width=True)

    # Gráfico de barras: número de pacientes por región
    df_barras = dff["region"].value_counts().reset_index()
    df_barras.columns = ["region", "pacientes"]
    fig_bar = px.bar(
        df_barras, x="region", y="pacientes",
        title=f"Número de pacientes por región (n={len(dff)})",
        color="region"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    # Título personalizado con n
    fig = px.scatter(
        dff, x="edad", y="gastos", color="fumador",
        size="imc", hover_data=["sexo", "region", "hijos"],
        title=f"Edad vs. gastos (tamaño del punto = IMC) (n={len(dff)})",
        color_discrete_map={"si": "#ea580c", "no": "#1e3a8a"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Título personalizado con n
    fig_imc = px.scatter(
        dff, x="imc", y="gastos", color="fumador",
        title=f"IMC vs. gastos (n={len(dff)})",
        color_discrete_map={"si": "#ea580c", "no": "#1e3a8a"},
        trendline="ols"
    )
    st.plotly_chart(fig_imc, use_container_width=True)

    # Gráfico de violín: comparar gastos entre hombres y mujeres separados por hábito de fumar
    fig_violin = px.violin(
        dff, x="sexo", y="gastos", color="fumador",
        title=f"Comparación de gastos por sexo y hábito de fumar (n={len(dff)})",
        color_discrete_map={"si": "#ea580c", "no": "#1e3a8a"},
        box=True
    )
    st.plotly_chart(fig_violin, use_container_width=True)

with tab3:
    st.dataframe(dff, use_container_width=True, height=400)
    csv = dff.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar datos filtrados (CSV)",
        csv, "gastos_filtrados.csv", "text/csv"
    )