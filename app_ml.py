"""
App de Machine Learning: Predicción de gastos médicos multi-modelo.
Curso de Ciencia de Datos - Ingeniería Industrial.
"""
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="Predicción de gastos médicos",
    page_icon="💰",
    layout="wide"
)

# ---------- Carga de modelos con caché ----------
@st.cache_resource
def cargar_modelos_y_metadata():
    """Carga los pipelines y las metadatas generadas en el entrenamiento."""
    # Cargamos las metadatas individuales para extraer R2 y RMSE en la sidebar
    meta_lineal = joblib.load("models/metadata_lineal.pkl")
    meta_arbol = joblib.load("models/metadata_arbol.pkl")
    meta_rf = joblib.load("models/metadata_rf.pkl")
    
    # Estructura de control interna para facilitar el manejo en la App
    modelos_dict = {
        "Regresión Lineal": {"pipeline": joblib.load("models/modelo_gastos.pkl"), "meta": meta_lineal},
        "Árbol de Regresión": {"pipeline": joblib.load("models/modelo_arbol.pkl"), "meta": meta_arbol},
        "Random Forest": {"pipeline": joblib.load("models/modelo_rf.pkl"), "meta": meta_rf}
    }
    return modelos_dict

modelos = cargar_modelos_y_metadata()

# ---------- Barra Lateral (Sidebar) ----------
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Selector solicitado para el modelo principal
    modelo_seleccionado = st.selectbox(
        "Selecciona el modelo principal:",
        ["Regresión Lineal", "Árbol de Regresión", "Random Forest"]
    )
    
    # Extraer pipeline y metadata del modelo elegido
    pipeline_principal = modelos[modelo_seleccionado]["pipeline"]
    meta_principal = modelos[modelo_seleccionado]["meta"]
    
    st.divider()
    st.header("📐 Calidad del modelo activo")
    st.metric("R² en test", f"{meta_principal['r2_test']:.3f}")
    st.metric("RMSE en test", f"${meta_principal['rmse_test']:,.0f}")

# ---------- Encabezado ----------
st.title("💰 Predicción de gastos médicos")
st.markdown(f"App interactiva para la evaluación y comparación de modelos predictivos.")

# ---------- Formulario de inputs ----------
with st.form("formulario_paciente"):
    st.subheader("📋 Datos del paciente")
    col1, col2 = st.columns(2)
    with col1:
        edad = st.number_input("Edad", 18, 64, 30)
        imc = st.number_input("IMC (Índice de Masa Corporal)", 15.0, 55.0, 25.0, 0.1)
        hijos = st.number_input("Número de hijos", 0, 5, 0)
    with col2:
        sexo = st.radio("Sexo", ["mujer", "hombre"], horizontal=True)
        fumador = st.radio("¿Es fumador?", ["no", "si"], horizontal=True)
        region = st.selectbox("Región", ["suroccidente", "suroriente", "noroccidente", "nororiente"])

    enviado = st.form_submit_button("🔮 Predecir gastos", type="primary", use_container_width=True)

# ---------- Predicción y resultados ----------
if enviado:
    # Construir DataFrame de entrada
    X_nuevo = pd.DataFrame([{"edad": edad, "imc": imc, "hijos": hijos, "sexo": sexo, "fumador": fumador, "region": region}])

    # =========================================================
    # PUNTO 4: COMPARACIÓN VISUAL LADO A LADO
    # =========================================================
    st.divider()
    st.subheader("📊 PUNTO 4: Comparación en tiempo real de los 3 modelos")
    
    pred_lineal = modelos["Regresión Lineal"]["pipeline"].predict(X_nuevo)[0]
    pred_arbol = modelos["Árbol de Regresión"]["pipeline"].predict(X_nuevo)[0]
    pred_rf = modelos["Random Forest"]["pipeline"].predict(X_nuevo)[0]
    
    comp1, comp2, comp3 = st.columns(3)
    with comp1:
        st.markdown("### Regresión Lineal")
        st.metric("Predicción", f"${pred_lineal:,.2f}")
        st.caption(f"RMSE: ${modelos['Regresión Lineal']['meta']['rmse_test']:,.0f}")
    with comp2:
        st.markdown("### Árbol de Regresión")
        st.metric("Predicción", f"${pred_arbol:,.2f}")
        st.caption(f"RMSE: ${modelos['Árbol de Regresión']['meta']['rmse_test']:,.0f}")
    with comp3:
        st.markdown("### Random Forest")
        st.metric("Predicción", f"${pred_rf:,.2f}")
        st.caption(f"RMSE: ${modelos['Random Forest']['meta']['rmse_test']:,.0f}")

    # Mostrar Gauge Chart del modelo principal seleccionado en la Sidebar
    st.divider()
    prediccion_principal = modelos[modelo_seleccionado]["pipeline"].predict(X_nuevo)[0]
    
    col_g1, col_g2 = st.columns([1, 2])
    with col_g1:
        st.subheader(f"Resultado Principal")
        st.metric(f"Estimación ({modelo_seleccionado})", f"${prediccion_principal:,.0f}")
    with col_g2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prediccion_principal,
            number={"prefix": "$", "valueformat": ",.0f"},
            title={"text": f"Gasto estimado con {modelo_seleccionado}"},
            gauge={"axis": {"range": [0, 50000]}, "bar": {"color": "#1e3a8a"}}
        ))
        fig.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # =========================================================
    # PUNTO 3: ADAPTAR LA SECCIÓN DE INTERPRETABILIDAD (IF/ELSE)
    # =========================================================
    with st.expander("🔍 PUNTO 3: ¿Cómo interpreta el modelo cada variable?"):
        algoritmo = pipeline_principal.named_steps["modelo"]
        nombres_features = pipeline_principal.named_steps["preprocesador"].get_feature_names_out()
        
        # Uso de la pista: type(modelo.named_steps["modelo"]).__name__
        tipo_modelo_str = type(algoritmo).__name__
        
        if tipo_modelo_str == "LinearRegression":
            # El modelo lineal usa coeficientes (pueden ser negativos o positivos)
            df_interp = pd.DataFrame({"variable": nombres_features, "valor": algoritmo.coef_})
            df_interp = df_interp.sort_values("valor", key=abs, ascending=False)
            titulo_grafico = "Coeficientes del Modelo Lineal (Naranja: Suma · Azul: Resta)"
            colores = ["#ea580c" if c > 0 else "#1e3a8a" for c in df_interp["valor"]]
            desc = "Muestra el impacto directo (positivo o negativo) en USD por cada unidad que cambia la variable."
        else:
            # Los árboles y ensamblados usan feature_importances_ (siempre positivos)
            df_interp = pd.DataFrame({"variable": nombres_features, "valor": algoritmo.feature_importances_})
            df_interp = df_interp.sort_values("valor", ascending=True)
            titulo_grafico = f"Importancia de Características ({tipo_modelo_str})"
            colores = ["#10b981"] * len(df_interp)
            desc = "Muestra la ganancia o peso relativo de la variable al reducir la varianza dentro del árbol (escala 0 a 1)."

        fig_interp = go.Figure(go.Bar(x=df_interp["valor"], y=df_interp["variable"], orientation="h", marker_color=colores))
        fig_interp.update_layout(title=titulo_grafico, height=350)
        st.plotly_chart(fig_interp, use_container_width=True)
        st.caption(desc)

# =========================================================
# PUNTO 5: RECOMENDACIÓN EN TEXTO (FUERA DEL IF DE PREDICCIÓN)
# =========================================================
st.divider()
with st.expander("📚 PUNTO 5: ¿Cuál modelo recomiendo?"):
    st.markdown("""
    ### 🏆 Recomendación Analítica para Producción
    
    Basado en el entrenamiento y validación cruzada del dataset de gastos médicos:
    
    1. **El Ganador en Rendimiento:** **Random Forest** suele presentar el **$R^2$ más alto** y el **$RMSE$ más bajo** en el conjunto de test. Esto ocurre porque captura de manera nativa las interacciones complejas y no lineales no estructuradas del dataset (como la penalización exponencial que sufren los pacientes fumadores que además registran un IMC superior a 30).
    2. **Riesgo de Sobreajuste (Overfitting):** Tal como indica la pista, los modelos basados en árboles tienden a memorizar el set de entrenamiento. Si bien el Random Forest reduce la varianza combinando múltiples estimadores, se debe vigilar que la brecha de métricas entre *train* y *test* no sea exagerada.
    3. **Decisión Final de Despliegue:** * Recomiendo usar **Random Forest en producción** debido a que los gastos médicos reales presentan dinámicas de comportamiento que un modelo puramente plano y lineal subestima severamente (provocando errores de predicción costosos). 
        * *Nota:* Si la interpretabilidad estricta, auditable y matemática fuera un requisito legal irrenunciable, la Regresión Lineal sería la alternativa obligada a pesar de su menor precisión.
    """)