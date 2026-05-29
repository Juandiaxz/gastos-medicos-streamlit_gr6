"""
Entrena múltiples modelos de regresión para predecir gastos médicos.
Ejecutar UNA VEZ con: python entrenamiento.py
"""
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

# 1. Cargar datos
df = pd.read_csv("data/gastos_medicos.csv")

# 2. Definir features y target
features_num = ["edad", "imc", "hijos"]
features_cat = ["sexo", "fumador", "region"]
target = "gastos"

X = df[features_num + features_cat]
y = df[target]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Definir el preprocesamiento común
preprocesador = ColumnTransformer([
    ("num", StandardScaler(), features_num),
    ("cat", OneHotEncoder(drop="first"), features_cat)
])

# Diccionario con los modelos a entrenar
modelos = {
    "lineal": {
        "instancia": LinearRegression(),
        "archivo": "models/modelo_gastos.pkl"
    },
    "arbol": {
        "instancia": DecisionTreeRegressor(max_depth=5, random_state=42),
        "archivo": "models/modelo_arbol.pkl"
    },
    "rf": {
        "instancia": RandomForestRegressor(n_estimators=100, random_state=42),
        "archivo": "models/modelo_rf.pkl"
    }
}

# Asegurar que el directorio de modelos exista
os.makedirs("models", exist_ok=True)

print("=" * 60)
print(f"{'EVALUACIÓN DE MODELOS':^60}")
print("=" * 60)

# 5. Entrenar, evaluar y guardar cada modelo
for nombre, configuracion in modelos.items():
    # Crear el pipeline específico para el modelo actual
    pipeline = Pipeline([
        ("preprocesador", preprocesador),
        ("modelo", configuracion["instancia"])
    ])
    
    # Entrenar
    pipeline.fit(X_train, y_train)
    
    # Predecir y evaluar
    y_pred_test = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = root_mean_squared_error(y_test, y_pred_test)
    
    # Reportar en consola
    print(f"📌 Modelo: {nombre.upper()}")
    print(f"   R² (test):   {r2:.4f}")
    print(f"   MAE (test):  ${mae:,.2f}")
    print(f"   RMSE (test): ${rmse:,.2f}")
    print("-" * 60)
    
    # Guardar el pipeline del modelo
    joblib.dump(pipeline, configuracion["archivo"])
    
    # Guardar metadata (sobrescribirá/creará una por cada modelo si lo deseas, 
    # aquí adaptado para guardar la metadata específica del modelo actual)
    metadata_archivo = f"models/metadata_{nombre}.pkl"
    joblib.dump({
        "features_num": features_num,
        "features_cat": features_cat,
        "rmse_test": rmse,
        "r2_test": r2
    }, metadata_archivo)

print("\n✅ ¡Todos los modelos y sus metadatos han sido guardados con éxito!")