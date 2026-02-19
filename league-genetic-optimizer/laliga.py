import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import copy
import random

# .\venv\Scripts\activate
# streamlit run laliga.py

# Funciones

def construir_jornada(indices_slots, partidos, slots_disponibles):
    # Inicializamos el diccionario vacío con los slots
    asignacion_jornada = {}
    for slot in slots_disponibles:
        asignacion_jornada[slot] = []
    
    # Llenamos el diccionario según los índices
    for i, idx_slot in enumerate(indices_slots):
        slot = slots_disponibles[idx_slot]
        partido = partidos[i]
        asignacion_jornada[slot].append(partido)
        
    return asignacion_jornada

def es_jornada_valida(asignacion_jornada):
    # No se va a comprobar si hay partidos en otros dias de la semana ya que se va a fijar que los partidos se juegen en esos días.
    # Definimos los días obligatorios, los especifico en vez de restar 4 por si en un futuro se cambian el día en el que se juegan los partidos.
    dias_requeridos = {"Viernes", "Sábado", "Domingo", "Lunes"}
    
    # Extraemos los días que tienen al menos un partido asignado
    dias_con_partidos = set()
    for (dia, hora), partidos in asignacion_jornada.items():
        if len(partidos) > 0:
            dias_con_partidos.add(dia)
    
    # Calculamos cuántos días de los obligatorios NO tienen partidos
    dias_faltantes = dias_requeridos - dias_con_partidos
    
    # Devolvemos el número entero de días que faltan (0, 1, 2, 3 o 4)
    return len(dias_faltantes)

def calcular_audiencia_partido(partido, dia, hora, num_partidos_en_slot):
    # Obtenmos audiencia base
    cat_local = equipo_categoria[partido[0]]
    cat_vis = equipo_categoria[partido[1]]
    base = audiencia_base[cat_local][cat_vis]
    
    # Aplicamos multiplicador de horario
    coef_horario = coeficientes_horarios.get((dia, hora), 1.0)
    audiencia_con_horario = base * coef_horario
    
    # Aplicamos penalización por coincidencia de partidos
    # coincidencias = total de partidos en ese slot menos el partido actual
    coincidencias = num_partidos_en_slot - 1
    reduccion = penalizaciones_coincidencia.get(coincidencias, 0.0)
    audiencia_final = audiencia_con_horario * (1 - reduccion)
    
    return audiencia_final

def fitness_final(indices_slots, lista_partidos,slots_disponibles):
    # Convertimos los indices de los partidos junto con la lista de partidos a el dicccionario de la jornada
    jornada = construir_jornada(indices_slots, lista_partidos,slots_disponibles)
        
    # Calculamos la audiencia total
    total_audiencia = 0
    for slot, partidos in jornada.items():
        dia, hora = slot
        num_partidos = len(partidos)
        for p in partidos:
            total_audiencia += calcular_audiencia_partido(p, dia, hora, num_partidos)
            
    # Comprobamos si es válida en función de los días y penalizamos si no lo es, al crear un corficiente que dismin en funcion del numero de dias que falten
    num_faltantes = es_jornada_valida(jornada)
    if num_faltantes == 0:
        return total_audiencia
    else:
        # En función del número de días faltantes se le penaliza en la audiencia total para que d+si en el torneo se encuentran el algoritmo sepa cual es el menos malo
        return total_audiencia - (num_faltantes * 5000000)

# predetermino 10 y 12 ya que en principio siempre tendrán estos valores
def generar_individuo_aleatorio(num_partidos=10, num_slots=12):
    individuo = []
    for i in range(num_partidos):
        slot_aleatorio = random.randint(0, num_slots - 1)
        individuo.append(slot_aleatorio)
    return individuo

def seleccionar_ganador_torneo(poblacion, lista_partidos, slots_disponibles, tamaño_torneo=5):
    # Elegimos unos cuantos individuos al azar de la población
    participantes_torneo = random.sample(poblacion, tamaño_torneo)
    
    # Buscamos quién es el mejor de esos participantes, para ello seleccionamos el priemro y comparamos el resto con el
    mejor_individuo = participantes_torneo[0]
    mejor_audiencia = fitness_final(mejor_individuo, lista_partidos, slots_disponibles)
    
    for individuo in participantes_torneo[1:]:
        audiencia = fitness_final(individuo, lista_partidos, slots_disponibles)
        
        if audiencia > mejor_audiencia:
            mejor_audiencia = audiencia
            mejor_individuo = individuo
            
    return mejor_individuo

def cruzar_padres(padre1, padre2):
    # Elegimos un punto de corte mitad
    punto_corte = len(padre1) // 2
    
    # El hijo toma la primera parte del padre1 y la segunda del padre2
    hijo = []
    
    # Primera mitad
    for i in range(0, punto_corte):
        hijo.append(padre1[i])
        
    # Segunda mitad
    for i in range(punto_corte, len(padre2)):
        hijo.append(padre2[i])

    return hijo

# Mutación 1: Cambiar un partido a un slot nuevo al azar
def mutacion_aleatoria(individuo, num_slots=12, probabilidad=0.05):
    for i in range(len(individuo)):
        if random.random() < probabilidad:
            individuo[i] = random.randint(0, num_slots - 1)
    return individuo

# Mutación 2: Intercambiar los horarios de dos partidos entre sí
def mutacion_intercambio(individuo, probabilidad=0.05):
    if random.random() < probabilidad:
        # Elegimos dos posiciones (partidos) al azar entre 0 y 9
        idx1 = random.randint(0, len(individuo) - 1)
        idx2 = random.randint(0, len(individuo) - 1)
        
        # Intercambiamos sus valores
        individuo[idx1], individuo[idx2] = individuo[idx2], individuo[idx1]
            
    return individuo

# DATOS Y RESTRICCIONES

EQUIPOS = [
    "Real Madrid CF", "Athletic Club", "Getafe CF", "Real Sociedad de Fútbol",
    "Girona FC", "Club Atlético Osasuna", "Villarreal CF", "Deportivo Alavés",
    "FC Barcelona", "Club Atlético de Madrid", "Real Oviedo", "Real Betis Balompié",
    "Sevilla FC", "RC Celta de Vigo", "Valencia CF", "Elche CF",
    "Levante UD", "RCD Espanyol de Barcelona", "Rayo Vallecano de Madrid", "RCD Mallorca"
]

# Matriz de Audiencias Base (en Millones) para el Sábado a las 22h
# Filas: Local, Columnas: Visitante
audiencia_base = {
    "A": {"A": 5.0, "B": 4.3, "C": 3.5}, 
    "B": {"A": 4, "B": 3.5, "C": 2.0}, 
    "C": {"A": 2.8, "B": 1.7, "C": 1.0}  
}

# Multiplicadores de audiencia por slot
# Formato: (Día, Hora): Coeficiente
coeficientes_horarios = {
    ("Viernes", "20:00"): 0.5, 
    ("Sábado", "12:00"): 0.55,
    ("Sábado", "16:00"): 0.7,
    ("Sábado", "18:00"): 0.9,
    ("Sábado", "20:00"): 0.9,
    ("Sábado", "22:00"): 1.0, # Referencia base
    ("Domingo", "12:00"): 0.6,
    ("Domingo", "16:00"): 0.9,
    ("Domingo", "18:00"): 0.7,
    ("Domingo", "20:00"): 0.8, 
    ("Domingo", "22:00"): 0.7, 
    ("Lunes", "20:00"): 0.5
}
# Convertimos las claves en una lista para tener un orden fijo un ejemplo sería el ("Viernes", "20:00") se convertiría en slots_disponibles[0], 
# esto lo hago para tenerlos por indice y asi aplicar las mutaciones variando solo los indices.
slots_disponibles = list(coeficientes_horarios.keys())

# Penalización por coincidencia de horarios
# Índice = número de coincidencias (además del propio partido)
# Como se ha mencionado en la función si hay 2 partidos a la vez, hay 1 coincidencia para cada uno.
penalizaciones_coincidencia = {
    0: 0.0,  # 0% reducción
    1: 0.25, # 25% reducción
    2: 0.40, # 40% reducción
    3: 0.50, # 50% reducción
    4: 0.60, # 60% reducción
    5: 0.65, # 65% reducción
    6: 0.70, # 70% reducción
    7: 0.75  # 75% reducción
}

# Diccionario directo: Equipo junto con su Categoría
equipo_categoria = {
    # Categoría A
    "Real Madrid CF": "A",
    "FC Barcelona": "A",
    "Club Atlético de Madrid": "A",

    # Categoría B
    "Athletic Club": "B",
    "Real Sociedad de Fútbol": "B",
    "Villarreal CF": "B",
    "Real Betis Balompié": "B",
    "Sevilla FC": "B",
    "Valencia CF": "B",
    "Girona FC": "B",
    "Getafe CF": "B",
    "RC Celta de Vigo": "B",
    "Rayo Vallecano de Madrid": "B",

    # Categoría C
    "RCD Mallorca": "C",
    "Club Atlético Osasuna": "C",
    "Deportivo Alavés": "C",
    "RCD Espanyol de Barcelona": "C",
    "Elche CF": "C",
    "Levante UD": "C",
    "Real Oviedo": "C"
}


# INTERFAZ DE USUARIO 
st.set_page_config(page_title="Optimizador de Jornadas LFP", layout="wide")
st.title("Optimizador de Horarios - Algoritmo Genético")
st.info(
    "En este archivo se realiza el **estudio y optimización de una única jornada**.\n\n"
    "Si desea realizar el **estudio de la liga completa y obtener el calendario más óptimo**, "
    "debe ejecutar el archivo **Page** disponible en el panel lateral izquierdo, debajo de este archivo.\n\n"
    "Si al ejecutar el algoritmo el gráfico no se actualiza esto se debe a que Streamlit está intentando actualizarse más rápido de lo que su navegador puede renderizar el gráfico simplemente espere, se actualizará en generaciones posteriores o " \
    "pulse el botón terminar y empiece otro; en el nuevo si que debería observarlo sin problemas desde el comienzo.\n\n"
    "La mutación 1 consiste en cambiar un partido a un nuevo horario al azar y la mutación 2 consiste en intercambiar los horarios de dos partidos entre sí."
)

col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Definir Partidos de la Jornada")
    
    # Se da a elegir al usuario si quiere poner el cada partido o si quiere que sea de forma aleatoria
    modo = st.radio("Modo de creación", ["Aleatorio", "Manual"])
    
    lista_equipos = list(equipo_categoria.keys())
    jornada_actual = []

    if modo == "Aleatorio":
        if st.button("Generar Pares Aleatorios"):
            equipos = lista_equipos.copy()   # Creamos una copia para no tocar el original por si fuera importante el orden que en este caso no pero por si acaso lo fuera
            random.shuffle(equipos) # Desordenar de manera aleatoria los elementos de una lista

            jornada_actual = []
            for i in range(0, len(equipos), 2):
                jornada_actual.append((equipos[i], equipos[i + 1]))

            st.session_state['j_jornada'] = jornada_actual

    else:
        # Si elige manual le apareceran todos los partidos paara que sellacciones a cada equipo en un deslizable, si selecciona un equipo en un partido con el número menor entonces
        #  se guardará en el nuevo partido y se eliminará el antiguo quedando ese lugar vacio
        equipos_en_uso = []
        for i in range(10):
            st.write(f"--- Partido {i+1} ---")
            c1, c2 = st.columns(2)
            
            # Bucle para filtrar equipos disponibles para el Local
            dispo_local = []
            for e in lista_equipos:
                if e not in equipos_en_uso:
                    dispo_local.append(e)
            
            loc = c1.selectbox(f"Local", ["-"] + dispo_local, key=f"l{i}")
            if loc != "-": 
                equipos_en_uso.append(loc)
            
            # Bucle para filtrar equipos disponibles para el Visitante
            dispo_visitante = []
            for e in lista_equipos:
                if e not in equipos_en_uso:
                    dispo_visitante.append(e)
            
            vis = c2.selectbox(f"Visitante", ["-"] + dispo_visitante, key=f"v{i}")
            if vis != "-": 
                equipos_en_uso.append(vis)
            
            if loc != "-" and vis != "-":
                jornada_actual.append((loc, vis))
        
        if len(equipos_en_uso) == 20:
            st.session_state['j_jornada'] = jornada_actual

with col2:
    st.header("2. Optimización por Algoritmo Genético")

    # Sliders para que se seleccione los parámetros para el modelo
    if 'j_jornada' in st.session_state:
        partidos = st.session_state['j_jornada']

        gen_max = st.slider("Generaciones Máximas", 50, 500, 200)
        pob_size = st.slider("Tamaño Población", 100, 2000, 500)
        paciencia = st.slider("Número de generaciones para evaluar si hay una mejora significativa", 5, 30, 15)
        mutacion1 = st.slider("Seleccione el porcentaje que tendrá la mutación 1 de suceder",0, 100, 10)
        mutacion2 = st.slider("Seleccione el porcentaje que tendrá la mutación 2 de suceder",0, 100, 10)

        #  se crean estados para que cuando el usuario de a finalizar el algoritmp no se vaya ni los gráficos ni las métricas, tendran nombres distintos respecto a la liga entera porque stramlit no diferencia que estado es de que archivo
        #  y los mezclaba dando error de primera uanque una vez le dabas a ejecutar iba sin problemas, pero para evitar que saltase ese error de primeras
        if 'j_ga_running' not in st.session_state:
            st.session_state.j_ga_running = False
        if 'j_gen_actual' not in st.session_state:
            st.session_state.j_gen_actual = 0
        if 'j_poblacion' not in st.session_state:
            st.session_state.j_poblacion = []
        if 'j_historial_audiencias' not in st.session_state:
            st.session_state.j_historial_audiencias = []
        if 'j_mejor_audiencia' not in st.session_state:
            st.session_state.j_mejor_audiencia = -1
        if 'j_mejor_calendario' not in st.session_state:
            st.session_state.j_mejor_calendario = None

        # Se crean los botones uno para iniciar y otro para finalizar
        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            if st.button("Ejecutar Algoritmo Genético"):
                # Al iniciar se cambia el estado de j_ga_running para que se cumplan las condiciones del bucle
                st.session_state.j_ga_running = True
                st.session_state.j_gen_actual = 0
                st.session_state.j_historial_audiencias = []
                st.session_state.j_mejor_audiencia = -1
                st.session_state.j_mejor_calendario = None
                st.session_state.j_poblacion = [
                    generar_individuo_aleatorio(len(partidos), len(slots_disponibles))
                    for _ in range(pob_size)
                ]
                st.rerun()

        with col_btn2:
            # Al detener se cambia el estado de j_ga_running para que no se cumplan las condiciones del bucle
            if st.button("Finalizar"):
                st.session_state.j_ga_running = False

        # Se crea el gráfico cuando ya haya un historial de audiencias sino lo hay saldrá un mensaje
        st.subheader("Evolución de la audiencia")
        if len(st.session_state.j_historial_audiencias) > 0:
            st.line_chart(
                pd.DataFrame(
                    {"Audiencia (M)": st.session_state.j_historial_audiencias}
                )
            )
        else:
            st.info("El gráfico se mostrará cuando comience la optimización.")

        # Se crean los huecos donde irán un mensaje advirtiendo la escasa mejoría y otro donde se mostrará la audiencia 
        metrics_placeholder = st.empty()
        warning_placeholder = st.empty()

        #  Empieza la iteración GA si j_ga_running esta en true y no se ha llegado a la última generación
        if st.session_state.j_ga_running and st.session_state.j_gen_actual < gen_max:
            
            # Se asocia los valores que el usuario ha elegido con las variables 
            poblacion = st.session_state.j_poblacion
            gen = st.session_state.j_gen_actual

            # Se crea una lista con la forma de bucle "comprimida" con el individuos y la audiencia (es el return de la funcion fitness_final)
            puntuados = [
                (ind, fitness_final(ind, partidos, slots_disponibles))
                for ind in poblacion
            ]
            # Se ordena de mayir a menor y se seleciona el primer elemento asociandolo como el mejor
            puntuados.sort(key=lambda x: x[1], reverse=True)

            mejor_actual, audiencia_actual = puntuados[0]

            # Si hay un historial de audiencias se mostrará la diferencia (delta) en porcentaje respecto a la mejor jornada anterior
            if len(st.session_state.j_historial_audiencias) == 0:
                delta_pct = None
            else:
                prev = st.session_state.j_historial_audiencias[-1]
                # Si la diferencia es positiva aparece en verde y si es negativa aprece e rojo (no va a aparecer en rojo nunca porque uso elitismo mas abajo al dejar pasar a alas dos con mejor audiencia)
                delta_pct = 100 * (audiencia_actual - prev) / prev if prev > 0 else None

            with metrics_placeholder:
                st.metric(
                    "Audiencia generación actual (M)",
                    f"{audiencia_actual:.2f}",
                    "–" if delta_pct is None else f"{delta_pct:.2f} %"
                )

            if audiencia_actual > st.session_state.j_mejor_audiencia:
                st.session_state.j_mejor_audiencia = audiencia_actual
                st.session_state.j_mejor_calendario = copy.deepcopy(mejor_actual)

            st.session_state.j_historial_audiencias.append(audiencia_actual)

            # Mensaje de Recomendación de parada si no se ha mejorado un 1% en 'generaciones' 
            if gen >= paciencia:
                aud_prev = st.session_state.j_historial_audiencias[gen - paciencia]
                if audiencia_actual <= aud_prev * 1.001:
                    warning_placeholder.warning(
                        f"La audiencia no ha mejorado significativamente "
                        f"en las últimas {paciencia} generaciones."
                    )

            # Se crea la Nueva población pasando los dos primeros y el resto se rellenan 1) realizando la seleccion por torneo de la funcion para selecionar a los padres 2) se cruzan 3) se mutan si se cumplen las condiciones
            nueva_pob = [puntuados[0][0], puntuados[1][0]]
            while len(nueva_pob) < pob_size:
                p1 = seleccionar_ganador_torneo(poblacion, partidos, slots_disponibles)
                p2 = seleccionar_ganador_torneo(poblacion, partidos, slots_disponibles)
                hijo = cruzar_padres(p1, p2)
                hijo = mutacion_aleatoria(hijo, len(slots_disponibles),mutacion1/100)
                hijo = mutacion_intercambio(hijo,mutacion2/100)
                nueva_pob.append(hijo)

            # Se actualiza el estado
            st.session_state.j_poblacion = nueva_pob
            st.session_state.j_gen_actual += 1
            st.rerun()

        # Una vez se finalize el modelo (por el usuario o que llega al máximo de iteraciones) se printean los Resultado final, que serán: la mejor audiencia total junto con la respectiva jornada
        if not st.session_state.j_ga_running and st.session_state.j_mejor_calendario is not None:

            st.success(
                f"Mejor audiencia alcanzada: "
                f"{st.session_state.j_mejor_audiencia:.2f} millones"
            )

            # Construimos la mejor jornada
            j_jornada_optima = construir_jornada(
                st.session_state.j_mejor_calendario,
                partidos,
                slots_disponibles
            )

            st.subheader("Jornada óptima obtenida")

            # Pasamos el diccionario a formato tabla
            filas = []
            for slot, games in j_jornada_optima.items():
                dia, hora = slot
                for g in games:
                    filas.append({
                        "Día": dia,
                        "Hora": hora,
                        "Partido": f"{g[0]} vs {g[1]}"
                    })

            if len(filas) > 0:
                st.table(pd.DataFrame(filas))
            else:
                st.warning("La jornada óptima no contiene partidos asignados.")
        
    else:
        st.info(
            "Primero debes definir la jornada en la **Fase 1**."
        )

