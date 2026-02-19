
# .\venv\Scripts\activate
# streamlit run practica.py

import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import pydeck as pdk
import plotly.graph_objects as go
import numpy as np
from streamlit_folium import st_folium
import folium
import joblib
from geopy.distance import geodesic
from lightgbm import LGBMRegressor
from category_encoders import TargetEncoder


# Función para asignar número máximo de personas según el tipo de habitación
def asignar_num_personas(room_type):
    if room_type == "Entire home/apt":
        return np.random.randint(2, 8)  # entre 2 y 7 personas
    elif room_type == "Private room":
        return np.random.randint(1, 4)  # entre 1 y 3 personas
    elif room_type == "Shared room":
        return np.random.randint(1, 5)  # entre 1 y 4 personas
    else:
        return np.nan  # por si aparece algún tipo nuevo no contemplado


# se carga en el caché para que no se recargue en cada iteración
@st.cache_data
def cargar_datos():
    # Fijam una semilla para que los valores no cambien en cada ejecución
    np.random.seed(42)
    data = pd.read_csv("AB_NYC_2019.csv")
    data["num_personas_max"] = data["room_type"].apply(asignar_num_personas)
    data.dropna(subset=['price', 'latitude', 'longitude'], inplace=True)
    return data

# Función que crea el mapa con todos los datos y tambien se guarda en el caché
@st.cache_resource
def generar_mapa(data):

    # Asignar colores por tipo de habitación
    color_map = {
        "Entire home/apt": [0, 128, 255],  # azul
        "Private room": [0, 255, 128],     # verde
        "Shared room": [255, 0, 128],      # rosa
    }

    # Se copia para no alterar la original y se le añade la columna color
    data = data.copy()
    data["color"] = data["room_type"].map(color_map)

    # Crear capa 3D
    layer = pdk.Layer(
        "ColumnLayer",
        data=data,
        get_position='[longitude, latitude]',
        get_elevation="price",
        elevation_scale=0.5,
        radius=40,
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
    )

    #  Configurar vista centrada en NYC con límites fijos
    view_state = pdk.ViewState(
        latitude=data["latitude"].mean(),
        longitude=data["longitude"].mean(),
        zoom=11,
        pitch= 50 # angulo de inclinación
        )

    #  Se carga el mapa elegido con los parámetros ya predeterminados
    pydeck_map = pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        layers=[layer],
        initial_view_state=view_state,
        tooltip={  # Añade una etiqueta que el usuario puede ver cuando clica en el 
            "html": (
                "<b>Nombre:</b> {name}<br>"
                "<b>Tipo:</b> {room_type}<br>"
                "<b>Precio:</b> {price} €<br>"
                "<b>Capacidad máxima:</b> {num_personas_max} personas<br>"
                "<b>Barrio:</b> {neighbourhood}<br>"
                "<b>Distrito:</b> {neighbourhood_group}"
            ),
            "style": {"backgroundColor": "steelblue", "color": "white"},
        }
        )

    return pydeck_map

# Función para recuperar el modelo entrenado para que no tenga que entrenarlo en cada iteración
@st.cache_resource
def cargar_modelo():
    ruta_modelo = "modelo_simulador.pkl"  
    modelo_guardado = joblib.load(ruta_modelo)
    return modelo_guardado

# Carga la base de datos
data = cargar_datos()
mapa = generar_mapa(data)

# Se crea el encabezado
st.set_page_config(page_title="Airbnb NYC 2019", layout="wide")
st.title("Sergio Mínguez Cruces — Airbnb NYC 2019")
st.markdown("## Resumen general de las características del dataset")

data.head()

# Métricas principales
# Creación de multiples columnas para facilitar la observación
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total de registros", f"{len(data):,}")
col2.metric("Precio medio", f"${data['price'].mean():.2f}")
col3.metric("Mínimo de noches promedio", f"{data['minimum_nights'].mean():.1f}")
col4.metric("Nº medio de personas", f"{data['num_personas_max'].mean():.1f}")

col5, col6, col7 = st.columns(3)
col5.metric("Número de barrios", data['neighbourhood_group'].nunique())
col6.metric("Número de distritos", data['neighbourhood'].nunique())
col7.metric("Tipos de habitación", data['room_type'].nunique())

# Se crea una seccion vacia que se rellenará cuando el df_filtered no este vacio
vacia = st.empty()

# Sidebar: Criterios de búsqueda
# Se crea una barra lateral en la que se pedirá al usuario que ingrese los criterios de búsqueda, por medio de depegables, sliders e inputs
st.sidebar.header("Criterios de búsqueda")

# despegables de barrio y distrito
neighbourhood_groups = data['neighbourhood_group'].unique()
group_sel = st.sidebar.selectbox("Selecciona un distrito (neighbourhood_group):", neighbourhood_groups)
df_filtered = data[data['neighbourhood_group'] == group_sel]

neighbourhoods = df_filtered['neighbourhood'].unique()
neighbourhood_sel = st.sidebar.selectbox("Selecciona un barrio (neighbourhood):", options=neighbourhoods)

# Desplegable múltiple de tipo de habitación
room_types = data['room_type'].unique()
room_sel = st.sidebar.multiselect("Selecciona el tipo de alojamiento (room_type):", room_types)

# Inputs de precio min max
min_price= st.sidebar.number_input("Precio mínimo en $:",step=1,min_value = 0)
max_price= st.sidebar.number_input("Precio máximo en $:",step=1,min_value = 0)
price_sel=[min_price,max_price]

# Slider del numero de personas
num_personas = st.sidebar.slider("Número de personas", min_value=1, max_value=7, value=2, step=1)

# Selector para diferenciar el tipo de duracion
tipo_estancia = st.sidebar.radio(
    "Tipo de estancia:",
    ("Corta duración (1-30 noches)", "Larga duración (31-365 noches)")
)

# Slider según tipo de estancia
if tipo_estancia == "Corta duración (1-30 noches)":
    min_nights_sel = st.sidebar.slider(
        "Mínimo de noches:",
        min_value=1,
        max_value=30,
        value=3
    )
else:
    min_nights_sel = st.sidebar.slider(
        "Mínimo de noches:",
        min_value=31,
        max_value=365,
        value=90
    )

# Creacion de las pestañas, con sus nombres asociados
tab1, tab2, tab3, tab4 = st.tabs(["Análisis general", "Gráficos comparativos","Exploración espacial", "Simulación de precios"])

with tab1:
    # PESTAÑA 1: Análisis general
    st.markdown("---")

    # Gráfico de distribución
    st.markdown("### Distribuciones clave")

    # Creación de dos columnas ya que los g´raficos no necesitan todo el ancho y se puede perfectamente dividir
    col1, col2 = st.columns(2)

    # Precio medio por tipo de habitación
    with col1:
        room_price = data.groupby("room_type")["price"].mean().reset_index().sort_values("price", ascending=False)
        fig1 = px.bar(room_price, x="room_type", y="price",
                    title="Precio medio por tipo de habitación",
                    color="room_type", text_auto=".2s")
        st.plotly_chart(fig1, use_container_width=True)

    # Precio medio por barrio
    with col2:
        neigh_price = data.groupby("neighbourhood_group")["price"].mean().reset_index().sort_values("price", ascending=False)
        fig2 = px.bar(neigh_price, x="neighbourhood_group", y="price",
                    title="Precio medio por barrio",
                    color="neighbourhood_group", text_auto=".2s")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Gráfico tipo de alojamiento vs numero de personas
    st.subheader("Distribución del número máximo de personas por tipo de habitación")

    fig = px.box(
        data,
        x="room_type",
        y="num_personas_max",
        color="room_type",
        labels={
            "room_type": "Tipo de habitación",
            "num_personas_max": "Número máximo de personas"
        },
        points="all"  # muestra los puntos individuales además del boxplot
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Mapa ubicaciones general
    st.markdown("### Distribución geográfica de alojamientos")
    st.pydeck_chart(mapa)
    st.color_picker("Entire home/apt", "#0080FF", label_visibility="visible", key= "color1")
    st.color_picker("Private room", "#00FF80", label_visibility="visible", key= "color2")
    st.color_picker("Shared room", "#FF0080", label_visibility="visible", key= "color3")

# PESTAÑA 4: Simulacion de precio con comparativa
with tab4:
    st.markdown("---")
    colA, colB = st.columns([1, 1])

    # Simulador de precio
    with colA:           

        st.subheader("Simulador de precios con modelo predictivo") 

        st.write("Introduce las características del alojamiento para estimar un precio aproximado:")

        # Inputs del usuario
        distrito_input = st.selectbox("Distrito:", sorted(data["neighbourhood_group"].unique()), key="sim_group")
        neigh_options = sorted(data[data["neighbourhood_group"] == distrito_input]["neighbourhood"].unique())
        barrio_input = st.selectbox("Barrio:", neigh_options, key="sim_neigh")
        room_input = st.selectbox("Tipo de alojamiento:", sorted(data["room_type"].unique()), key="sim_room")

        min_nights_input = st.slider("Número mínimo de noches:", 1, 30, 3, key="sim_min_nights")
        personas_input = st.slider("Número de personas:",1,max(data["num_personas_max"]),2,key="sim_personas")
        
        subset = data[
            (data["neighbourhood_group"] == distrito_input) &
            (data["neighbourhood"] == barrio_input)
                    ]
        
        st.markdown("### Selecciona la ubicación en el mapa")

        #Límites y centro de la latitud y longitud del barrio
        lat_min, lat_max = subset["latitude"].min(), subset["latitude"].max()
        lon_min, lon_max = subset["longitude"].min(), subset["longitude"].max()
        lat_center = (lat_min + lat_max) / 2
        lon_center = (lon_min + lon_max) / 2

        # Crear el mapa centrado en el barrio y creación del cuadrdado para indicar al usuario el barrio
        m = folium.Map(location=[lat_center, lon_center], zoom_start=14) 
        folium.Rectangle(
            bounds=[[lat_min, lon_min], [lat_max, lon_max]],
            color="blue", fill=True, fill_opacity=0.1
        ).add_to(m)
        folium.LatLngPopup().add_to(m)

        # Mostrar el mapa y guardar el click
        map_output = st_folium(m, width=700, height=500)

        # Se inicieliza la latitud y longitud del click en None por si el usuario no pulsara el mapa
        lat_click, lon_click = None, None
        if map_output["last_clicked"] is not None:
            lat_click = map_output["last_clicked"]["lat"]
            lon_click = map_output["last_clicked"]["lng"]
            st.success(f"Ubicación seleccionada: lat={lat_click:.5f}, lon={lon_click:.5f}")

        if lat_click != None:
            if ((lat_min >= lat_click) or (lat_click >= lat_max) or (lon_min >= lon_click) or  (lon_click >= lon_max)):
                st.warning("La ubicación seleccionada está fuera del barrio. Se usará el centro del área.")
                lat_click, lon_click = lat_center, lon_center

        

        # Botón para entrenar modelo
        entrenar = st.button("Estimar precio")

        if entrenar:

            # Cargar el modelo y se extrae las caracteríaticas que se usarán para el prepocesado
            modelo_dict = cargar_modelo()
            modelo = modelo_dict["modelo"]
            encoder = modelo_dict["encoder"]
            scaler = modelo_dict["scaler"]
            features = modelo_dict["features"]
            categoricas = modelo_dict["categoricas"]
            numericas = modelo_dict["numericas"]

            # Si no hubiese clicado en el mapa se toma la media
            if lat_click is None or lon_click is None:
                st.warning("No se seleccionó una ubicación. Se usará la media del barrio.")
                lat_click = subset["latitude"].mean()
                lon_click = subset["longitude"].mean()
            
            # Se crea el df en el que recoge los inputs del usuario para el prepocesamiento, se inicializa dist_center ya que es necesaria para el modelo
            input_data = pd.DataFrame({
                'neighbourhood_group': [distrito_input],
                'neighbourhood': [barrio_input],
                'room_type': [room_input],
                'minimum_nights': [min_nights_input],
                'num_personas_max': [personas_input],
                'latitude': [lat_click],
                'longitude' : [lon_click],
                'dist_centro': [0]
            })

            center = (40.758, -73.9855) # la ubicación de Times Square
            # Se actualiza la distacia al centro
            input_data["dist_centro"] = input_data.apply(
                lambda row: geodesic(center, (row.latitude, row.longitude)).km, axis=1
            )

            # Codificar variables categóricas igual que en entrenamiento
            input_data[categoricas] = encoder.transform(input_data[categoricas])

            # Escalar variables numéricas igual que en entrenamiento
            input_data[numericas] = scaler.transform(input_data[numericas])

            pred_log = modelo.predict(input_data)[0]
            precio_estimado = np.expm1(pred_log)  # revertir log1p a la variable price

            st.metric(label="Precio estimado por noche", value=f"${precio_estimado:.2f}")


    # Graficos de las comparaciones
    with colB:
        if entrenar:
            # Gráfico del barrio 
            st.subheader("Comparativa de precios con el barrio")

            df_group = data[
                (data["neighbourhood_group"] == distrito_input) &
                (data["room_type"] == room_input) &
                (data["neighbourhood"] == barrio_input) &
                (data['minimum_nights'] >= min_nights_input) &
                (data['num_personas_max'] >= personas_input)
            ]

            if df_group.empty:
                st.warning("No hay otras habitaciones que cumplan tus características en el distrito.")
            else:
                p90 = df_group["price"].quantile(0.90)
                p50 = df_group["price"].median()
                x_max = max(p90, p50, precio_estimado) * 1.2  # Para que la etiqueta se vea

                data_comp = pd.DataFrame({
                    "Categoría": ["Top 10 %", "Media", "Tu simulación"],
                    "Precio": [p90, p50, precio_estimado],
                    "Color": ["#2ecc71", "#95a5a6", "#3498db"]
                })

                altair_fig = (
                    alt.Chart(data_comp)
                    .mark_bar(size=40)
                    .encode(
                        x=alt.X("Precio:Q", title="Precio (USD)", scale=alt.Scale(domain=[0, x_max])),
                        y=alt.Y("Categoría:N", sort=["Top 10 %", "Media", "Tu simulación"]),
                        color=alt.Color("Color:N", scale=None),
                        tooltip=["Categoría", "Precio"]
                    )
                    .properties(width=600, height=300)
                )

                text = alt.Chart(data_comp).mark_text(
                    align="left", baseline="middle", dx=5
                ).encode(
                    x="Precio:Q", y="Categoría:N",
                    text=alt.Text("Precio:Q", format=".0f")
                )

                grafico_comp = altair_fig + text
                st.altair_chart(grafico_comp, use_container_width=True)

            st.markdown("---")
            # Gráfico de distritos
            st.subheader("Comparativa de precios con el distrito")

            df_group = data[
                (data["neighbourhood_group"] == distrito_input) &
                (data["room_type"] == room_input) &
                (data["minimum_nights"] >= min_nights_input) &
                (data['num_personas_max'] >= personas_input)
            ]

            if df_group.empty:
                st.warning("No hay otras habitaciones que cumplan tus características en el distrito.")
            else:
                p90 = df_group["price"].quantile(0.90)
                p50 = df_group["price"].median()

                # Aumentar el límite del eje X un 20% por encima del máximo
                x_max = max(p90, p50, precio_estimado) * 1.2

                data_comp = pd.DataFrame({
                    "Categoría": ["Top 10 %", "Media", "Tu simulación"],
                    "Precio": [p90, p50, precio_estimado],
                    "Color": ["#2ecc71", "#95a5a6", "#3498db"]
                })

                altair_fig = (
                    alt.Chart(data_comp)
                    .mark_bar(size=40)
                    .encode(
                        x=alt.X("Precio:Q", title="Precio (USD)", scale=alt.Scale(domain=[0, x_max])),
                        y=alt.Y("Categoría:N", sort=["Top 10 %", "Media", "Tu simulación"]),
                        color=alt.Color("Color:N", scale=None),
                        tooltip=["Categoría", "Precio"]
                    )
                    .properties(width=600, height=300)
                )

                text = alt.Chart(data_comp).mark_text(
                    align="left", baseline="middle", dx=5
                ).encode(
                    x="Precio:Q", y="Categoría:N",
                    text=alt.Text("Precio:Q", format=".0f")
                )

                grafico_comp = altair_fig + text
                st.altair_chart(grafico_comp, use_container_width=True)
        
        else:
            st.info("Selecciona las características y pulsa **'Estimar precio'** para poder ver la comparativa.")
            

# Filtro principal de datos

df_filtered = df_filtered[df_filtered['room_type'].isin(room_sel)]
df_filtered = df_filtered[df_filtered['price'].between(*price_sel)]
df_filtered = df_filtered[df_filtered['minimum_nights'] >= min_nights_sel]
df_filtered = df_filtered[df_filtered['neighbourhood'] == neighbourhood_sel]
df_filtered = df_filtered[df_filtered['num_personas_max'] >= num_personas]

# Verificar si el DataFrame resultante está vacío
if df_filtered.empty:
    with vacia.container():
        st.warning("No hay resultados con los filtros seleccionados. Prueba ajustando los parámetros.")

    with tab2:
        st.warning("No hay resultados con los filtros seleccionados. Prueba ajustando los parámetros.")

    with tab3:
        st.warning("No hay resultados con los filtros seleccionados. Prueba ajustando los parámetros.")
else:
    with vacia.container():
        st.write(f"De los {len(data):,} alojamientos disponibles, {len(df_filtered):,} cumplen tus características.")
        st.write("Si quieres ubicarlos en el mapa, selecciona el apartado *Exploración espacial*.")
        st.dataframe(df_filtered[[
            "name", "host_id", "neighbourhood_group", "neighbourhood",
            "room_type", "price", "minimum_nights", "num_personas_max"
        ]])

    with tab2:
        # PESTAÑA 2 — Gráficos comparativa
        st.markdown("---")
        # Gráfico tipo de alojamiento vs mínimo de noches
        st.subheader("Relación entre tipo de alojamiento y noches mínimas requeridas")

        fig1 = px.scatter(
        df_filtered,
        x="minimum_nights",
        y="price",
        color="number_of_reviews",
        symbol="room_type",
        hover_data=["name", "neighbourhood"], # Informacion adicional que aparece al pasarle el raton
        color_continuous_scale="Viridis"
    )

        # Mejoras estéticas
        fig1.update_traces(
            marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color="DarkSlateGrey"))
        )

        fig1.update_layout(
            xaxis_title="Noches mínimas",
            yaxis_title="Precio (€)",
            coloraxis_colorbar=dict(title="Nº de reseñas"),
            legend_title_text="Tipo de alojamiento",
            legend=dict(
                yanchor="top",
                y=0.98, # mueve la leyenda en el eje y
                xanchor="left",
                x=0.89,  # mueve la leyenda en el eje x
                bgcolor="rgba(255,255,255,0.6)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            ),
            margin=dict(r=120),  # deja espacio a la derecha
            template="plotly_white"
        )

        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("---")
        # Gráfico relación entre disponibilidad anual y precio
        st.subheader("Relación entre disponibilidad anual y precio")

        fig2 = px.scatter(
            df_filtered,
            x="availability_365",
            y="price",
            color="room_type",
            hover_data=["name", "neighbourhood", "room_type"],
            trendline="ols"
        )

        # Ajustes visuales
        fig2.update_traces(
            marker=dict(size=7, opacity=0.6, line=dict(width=0.3, color="DarkSlateGrey"))
        )

        fig2.update_layout(
            xaxis_title="Disponibilidad anual (días)",
            yaxis_title="Precio (€)",
            legend_title_text="Distrito",
            template="plotly_white"
        )

        st.plotly_chart(fig2, use_container_width=True)


    # PESTAÑA 3: Exploración espacial

    with tab3:
        st.markdown("---")
        col1, col2 = st.columns([1, 1.2]) # ancho de las columnas la segunda columna es más ancha porque el mapa al ser solo del baarrio no hacaae falta que se tan anco y asi se observa mejor el otro gráfico

        with col1:

            # Mapa 3D precio y tipo de habitación
            st.markdown(f"### Distribución geográfica de alojamientos en el barrio **{neighbourhood_sel}**")

            # Asegurar de que las columnas sean del tipo adecuado
            df_filtered["latitude"] = df_filtered["latitude"].astype(float)
            df_filtered["longitude"] = df_filtered["longitude"].astype(float)
            df_filtered["price"] = df_filtered["price"].astype(float)

            color_map = {
                "Entire home/apt": [0, 128, 255],    # azul
                "Private room": [0, 255, 128],       # verde
                "Shared room": [255, 0, 128],        # rosa
            }

            df_filtered["color"] = df_filtered["room_type"].map(color_map)

            # Definir la capa de PyDeck
            layer = pdk.Layer(
                "ColumnLayer",
                data=df_filtered,
                get_position='[longitude, latitude]',
                get_elevation="price",
                elevation_scale=0.5,     # Ajusta la escala vertical
                radius=40,               # Tamaño de cada columna
                get_fill_color="color",  # Color según tipo de habitación
                pickable=True,
                auto_highlight=True,
            )

            # Configuración del mapa base 
            view_state = pdk.ViewState(
                latitude=df_filtered["latitude"].mean(),
                longitude=df_filtered["longitude"].mean(),
                zoom=13,
                pitch= 50
                 
            )

            # Crear el mapa
            pydeck_map = pdk.Deck(
                map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
                layers=[layer],
                initial_view_state=view_state,
                tooltip={
                    "html": "<b>Nombre:</b> {name}<br>"
                            "<b>Tipo:</b> {room_type}<br>"
                            "<b>Precio:</b> {price} €<br>"
                            "<b>Capacidad máxima:</b> {num_personas_max} personas<br>",
                    "style": {"backgroundColor": "steelblue", "color": "white"}
                }
            )

            # Mostramos el mapa
            st.pydeck_chart(pydeck_map)

            # Leyenda 
            st.color_picker("Entire home/apt", "#0080FF", label_visibility="visible")
            st.color_picker("Private room", "#00FF80", label_visibility="visible")
            st.color_picker("Shared room", "#FF0080", label_visibility="visible")
           

        with col2:
            
           # Distribución por barrio
            st.subheader(f"Distribución de tipos de alojamiento en el barrio **{neighbourhood_sel}**")

            df_filtered3 = data[data['neighbourhood_group'] == group_sel]
            df_filtered3 = df_filtered3[df_filtered3['price'].between(*price_sel)]
            df_filtered3 = df_filtered3[df_filtered3['minimum_nights'] >= min_nights_sel]
            df_filtered3 = df_filtered3[df_filtered3['neighbourhood'] == neighbourhood_sel]

            df_tipo = df_filtered3["room_type"].value_counts().reset_index()
            df_tipo.columns = ["room_type", "count"]

            fig_barrio = px.bar(
                df_tipo,
                x="room_type",
                y="count",
                color="room_type",
                text="count",
                labels={"room_type": "Tipo de habitación", "count": "Número de alojamientos"}
            )

            fig_barrio.update_traces(texttemplate='%{text}', textposition='outside')
            ymax_barrio = df_tipo["count"].max()
            fig_barrio.update_layout(yaxis_range=[0, ymax_barrio * 1.25], margin=dict(t=50, b=40))
            st.plotly_chart(fig_barrio, use_container_width=True)

            st.markdown("---")

            # Distribución por distrito
            st.subheader(f"Distribución de tipos de alojamiento en el distrito **{group_sel}**")

            df_filtered2 = data[data['neighbourhood_group'] == group_sel]
            df_filtered2 = df_filtered2[df_filtered2['price'].between(*price_sel)]
            df_filtered2 = df_filtered2[df_filtered2['minimum_nights'] >= min_nights_sel]

            df_tipo2 = df_filtered2["room_type"].value_counts().reset_index()
            df_tipo2.columns = ["room_type", "count"]

            fig_distrito = px.bar(
                df_tipo2,
                x="room_type",
                y="count",
                color="room_type",
                text="count",
                labels={"room_type": "Tipo de alojamiento", "count": "Número de alojamientos"}
            )

            # Etiqueta con el valor maximo y aumenta la altura del eje para que se vea la etiqueta
            fig_distrito.update_traces(texttemplate='%{text}', textposition='outside')
            ymax_distrito = df_tipo2["count"].max()
            fig_distrito.update_layout(yaxis_range=[0, ymax_distrito * 1.25], margin=dict(t=50, b=40))
            st.plotly_chart(fig_distrito, use_container_width=True)

    