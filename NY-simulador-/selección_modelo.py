import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from geopy.distance import geodesic
from category_encoders import TargetEncoder

# Carga y prepocesamiento de los datos
data = pd.read_csv("AB_NYC_2019.csv")
data.dropna(subset=['price', 'latitude', 'longitude'], inplace=True)

# Eliminar outliers por rango intercuartílico (IQR) 
Q1, Q3 = data['price'].quantile([0.25, 0.75])
IQR = Q3 - Q1
limite_inf, limite_sup = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
data = data[(data['price'] >= limite_inf) & (data['price'] <= limite_sup)]

np.random.seed(42)
def asignar_num_personas(room_type):
    if room_type == "Entire home/apt":
        return np.random.randint(2, 8)
    elif room_type == "Private room":
        return np.random.randint(1, 4)
    elif room_type == "Shared room":
        return np.random.randint(1, 5)
    else:
        return 1
    
data["num_personas_max"] = data["room_type"].apply(asignar_num_personas)

# Añadir distancia a Times Square ya que la distancia al centro suele ser motivo de aumento del precio
center = (40.758, -73.9855)
data["dist_centro"] = data.apply(
    lambda row: geodesic(center, (row.latitude, row.longitude)).km, axis=1
)

# Separación de variables explicativas
features = [
    'neighbourhood_group', 'neighbourhood', 'room_type',
    'minimum_nights', 'num_personas_max', 'latitude', 'longitude', 'dist_centro'
]
target = 'price'

categoricas = ['neighbourhood_group', 'neighbourhood', 'room_type']
numericas = ['minimum_nights', 'num_personas_max', 'latitude', 'longitude', 'dist_centro']

# Procesamiento de las variables seleccionadas
te = TargetEncoder(cols=categoricas) # vuelve las catagóricas a numéricas para poder aplicar los modelos
X = data[features].copy()
y = np.log1p(data[target])  # log-transform para estabilidad
X[categoricas] = te.fit_transform(X[categoricas], y)

scaler = StandardScaler() 
X[numericas] = scaler.fit_transform(X[numericas])

# Dividir el conjunto en train y test 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Modelos a entrenar
modelos = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=40, random_state=42)
}

# Lista donde se guadarán las métricas de los modeloes
resultados = []

# Bucle donde se entrenarán los modelos 
for nombre, modelo in modelos.items():
    print(f"\nEntrenando {nombre} ...")
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # Métricas del modelo
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    resultados.append((nombre, r2, mae))

# Se muestran los resultados por orden en función del R^2
df_result = pd.DataFrame(resultados, columns=["Modelo", "R^2 (test)", "MAE (test)"])
print("\nTabla comparativa")
print(df_result.sort_values(by="R^2 (test)", ascending=False))


