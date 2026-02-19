import streamlit as st
import random
import pandas as pd
import copy


def generar_calendario_base_liga(equipos):
    # Sé que son 20 pero por si acaso hay expnasión de la liga
    n = len(equipos)
    if n % 2 != 0:
        return None # Siempre debe ser par para que no haya uno que se quede descansando

    # Dividimos la lista para la rotación
    fijo = equipos[0]
    rotatorios = equipos[1:]
    
    jornadas_ida = []

    # Generamos la Primera Vuelta (19 jornadas) 
    for i in range(n - 1):
        jornada = []
        # El primer partido es el fijo vs el último de los rotatorios
        jornada.append((fijo, rotatorios[-1]))
        
        # El resto de parejas se forman cruzando el inicio con el final de equipo en rotatorios
        for j in range((n // 2) - 1):
            jornada.append((rotatorios[j], rotatorios[-(j + 2)]))
        
        jornadas_ida.append(jornada)
        
        # Rotamos la lista: el último pasa a ser el primero
        rotatorios = [rotatorios[-1]] + rotatorios[:-1]

    # Generamos la Segunda Vuelta (que consiste en invertir la localidad de la primera vuelta) 
    jornadas_vuelta = []
    for j_ida in jornadas_ida:
        jornada_v = []
        for partido in j_ida:
            # Si en la ida fue (A, B), en la vuelta es (B, A)
            jornada_v.append((partido[1], partido[0]))
        jornadas_vuelta.append(jornada_v)

    # Combinamos ambas vueltas (38 jornadas en total)
    return jornadas_ida + jornadas_vuelta

def generar_individuo_liga_completa(mapa_maestro, slots_disponibles):
    # Obtenemos los números automáticamente del mapa_maestro
    num_jornadas = len(mapa_maestro)
    partidos_por_jornada = len(mapa_maestro[0]) # Asumimos que todas tienen los mismos (que son 10 hasta expansión de la liga)
    num_slots = len(slots_disponibles)
    
    individuo = []
    for j in range(num_jornadas):
        jornada_horarios = []
        for p in range(partidos_por_jornada):
            # Asignación totalmente aleatoria (puede no ser válida)
            jornada_horarios.append(random.randint(0, num_slots - 1))
        individuo.append(jornada_horarios)
    return individuo

def es_jornada_valida(asignacion_jornada):
    # No se va a comprobar si hay partidos en otros dias de la semana (tipo martes).
    # Definimos los días obligatorios, los especifico en vez de restar 4 por si en un futuro se cambian el día en el que se juegan los partidos.
    dias_requeridos = {"Viernes", "Sábado", "Domingo", "Lunes"}
    
    # Extraemos los días que tienen al menos un partido asignado
    dias_con_partidos = set()
    for (dia, hora), partidos in asignacion_jornada.items():
        if len(partidos) > 0:
            dias_con_partidos.add(dia)
    
    # Comprobamos si faltan días
    dias_faltantes = dias_requeridos - dias_con_partidos
    
    if not dias_faltantes:
        return True
    else:
        return False

def calcular_audiencia_partido(partido, dia, hora, num_partidos_en_slot):
    # Obtenmos audiencia base
    cat_local = equipo_categoria[partido[0]]
    cat_vis = equipo_categoria[partido[1]]
    base = audiencia_base[cat_local][cat_vis]
    
    
    # Aplicamos multiplicador de horario
    # Usamos .get() por seguridad, con 1.0 como valor por defecto
    coef_horario = coeficientes_horarios.get((dia, hora), 1.0)
    audiencia_con_horario = base * coef_horario
    
    # Aplicamos penalización por coincidencia de partidos
    # coincidencias = total de partidos en ese slot menos el partido actual
    coincidencias = num_partidos_en_slot - 1
    reduccion = penalizaciones_coincidencia.get(coincidencias, 0.0)
    audiencia_final = audiencia_con_horario * (1 - reduccion)
    
    return audiencia_final

def calcular_audiencia_jornada(partidos_jornada, indices_horarios):
    # Necesitamos saber cuántos partidos hay por cada slot para las penalizaciones, se crea un diccionario donde cada clave es el indice y el valor el numero de coincidencias
    conteo_slots = {}
    for idx_slot in indices_horarios:
        if idx_slot not in conteo_slots:
            conteo_slots[idx_slot] = 0
        conteo_slots[idx_slot] += 1
    
    audiencia_total_jornada = 0
    
    # Calculamos la audiencia partido a partido y las sumamos todas para saber la de la jornada
    for i in range(len(partidos_jornada)):
        partido = partidos_jornada[i]
        idx_slot = indices_horarios[i]
        
        # Obtenemos el nombre del día y hora desde tu lista global de slots
        dia, hora = slots_disponibles[idx_slot]
        
        # Usamos tu función de cálculo de partido que ya tienes
        num_coincidencias = conteo_slots[idx_slot]
        aud = calcular_audiencia_partido(partido, dia, hora, num_coincidencias)
        audiencia_total_jornada += aud
        
    return audiencia_total_jornada

def fitness_liga_completa(individuo_horarios, calendario_maestro, slots_disponibles):
    audiencia_total_liga = 0
    conteo_validas = 0
    num_jornadas = len(calendario_maestro)
    for j in range(num_jornadas):
        indices_horarios = individuo_horarios[j]
        # Se valida si la jornada el válida, si lo es se aumenta el valor de conteo_validas
        dias_en_jornada = {slots_disponibles[idx][0] for idx in indices_horarios}
        es_valida = {"Viernes", "Sábado", "Domingo", "Lunes"}.issubset(dias_en_jornada)
        
        if es_valida:
            conteo_validas += 1
               
        partidos_jornada = calendario_maestro[j]
        audiencia_total_liga += calcular_audiencia_jornada(partidos_jornada, indices_horarios)

    # Se calcula una varible fitnes que será la que equilibre entre la audiencia y la validez de la jornadas
    fitness = audiencia_total_liga + (conteo_validas * 5.0)
    return fitness, audiencia_total_liga, conteo_validas

def cruzar_padres_liga(padre1, padre2):
    # Elegimos un punto de corte entre las 38 jornadas
    punto_corte = random.randint(1, 37)
    
    # El hijo toma las primeras jornadas del padre1 y el resto del padre2
    hijo = padre1[:punto_corte] + padre2[punto_corte:]
    
    return hijo

def mutacion_liga(individuo_liga, slots_disponibles, prob=0.05):
    # Identificamos qué slots pertenecen a qué día para que la mutación no invalide jornadas validas
    indices_por_dia = {"Viernes": [], "Sábado": [], "Domingo": [], "Lunes": []}
    for i, slot in enumerate(slots_disponibles):
        indices_por_dia[slot[0]].append(i)

    for j in range(len(individuo_liga)):
        if random.random() < prob:
            jornada = individuo_liga[j]
            idx_partido = random.randint(0, 9)
            
            # Guardamos el valor antiguo por si acaso
            antiguo_slot = jornada[idx_partido]
            nuevo_slot = random.randint(0, len(slots_disponibles) - 1)
            
            jornada[idx_partido] = nuevo_slot
            
            # Se comprueba la validez:
            # Si al cambiarlo la jornada deja de ser válida, deshacemos la mutación
            dias_presentes = {slots_disponibles[s][0] for s in jornada}
            dias_obligatorios = {"Viernes", "Sábado", "Domingo", "Lunes"}
            
            if not dias_obligatorios.issubset(dias_presentes):
                jornada[idx_partido] = antiguo_slot 
                
    return individuo_liga

def seleccionar_ganador_torneo_liga(poblacion, mapa_liga, slots_disponibles, tamaño_torneo=5):
    # Elegimos unos cuantos calendarios al azar
    participantes_torneo = random.sample(poblacion, tamaño_torneo)
    
    mejor_individuo = None
    mejor_fitness = -1
    mejor_audiencia = -1
    mejor_validez = -1
    
    for individuo in participantes_torneo:
        res_fitness = fitness_liga_completa(individuo, mapa_liga, slots_disponibles)
        
        # Como la función devuelve (fitness, audiencia, validas), extraemos la audiencia y fitness para que el ganador de prioridad al quilibrio de las jornadas, después al número de audiencia y por último la validez de las jornadas
        # lo he hecho así ya que si quedan pocas jornadas simplemente se podría usar la optimización de una única jornada para las 2 o 3 que quedarán.
        fitness = res_fitness[0] 
        audiencia = res_fitness[1] 
        validez = res_fitness[2]
        
        if fitness > mejor_fitness:
            mejor_fitness = fitness
            mejor_individuo = individuo
            mejor_audiencia = audiencia
            mejor_validez = validez

        elif fitness == mejor_fitness and audiencia > mejor_audiencia:
            mejor_fitness = fitness
            mejor_individuo = individuo
            mejor_audiencia = audiencia
            mejor_validez = validez

        elif fitness == mejor_fitness and audiencia == mejor_audiencia and validez > mejor_validez:
            mejor_fitness = fitness
            mejor_individuo = individuo
            mejor_audiencia = audiencia
            mejor_validez = validez
            
    return mejor_individuo

def construir_tabla_liga(individuo_ganador, mapa_liga, slots_disponibles):
    filas = []
    for j in range(38):
        indices_j = individuo_ganador[j]
        partidos_j = mapa_liga[j]
        
        for i in range(10):
            idx_slot = indices_j[i]
            dia, hora = slots_disponibles[idx_slot]
            local, visitante = partidos_j[i]
            
            filas.append({
                "Jornada": j + 1,
                "Partido": f"{local} vs {visitante}",
                "Día": dia,
                "Hora": hora,
                "Categoría": f"{equipo_categoria[local]}-{equipo_categoria[visitante]}"
            })
    return filas

EQUIPOS = [
    "Real Madrid CF", "Athletic Club", "Getafe CF", "Real Sociedad de Fútbol",
    "Girona FC", "Club Atlético Osasuna", "Villarreal CF", "Deportivo Alavés",
    "FC Barcelona", "Club Atlético de Madrid", "Real Oviedo", "Real Betis Balompié",
    "Sevilla FC", "RC Celta de Vigo", "Valencia CF", "Elche CF",
    "Levante UD", "RCD Espanyol de Barcelona", "Rayo Vallecano de Madrid", "RCD Mallorca"
]

# Diccionario de Audiencias Base (en Millones) para el Sábado 22h
# Clave: Local, valores: diccionarios donde la subclave es el visitante y el valor es la audiencia del partido
audiencia_base = {
    "A": {"A": 5.0, "B": 4.3, "C": 3.5}, 
    "B": {"A": 4.0, "B": 3.5, "C": 2.0}, 
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
# esto lo hago para tenerlos por indice y asi aplicar las mutaciones variando solo los indices
slots_disponibles = list(coeficientes_horarios.keys())

# Penalización por coincidencia de horarios
# Índice = número de coincidencias (además del propio partido)
# Si hay 2 partidos a la vez, hay 1 coincidencia para cada uno.
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

# Diccionario directo: Equipo -> Categoría
equipo_categoria = {
    # --- Categoría A ---
    "Real Madrid CF": "A",
    "FC Barcelona": "A",
    "Club Atlético de Madrid": "A",

    # --- Categoría B ---
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

    # --- Categoría C ---
    "RCD Mallorca": "C",
    "Club Atlético Osasuna": "C",
    "Deportivo Alavés": "C",
    "RCD Espanyol de Barcelona": "C",
    "Elche CF": "C",
    "Levante UD": "C",
    "Real Oviedo": "C"
}

# INTERFAZ DE USUARIO 
st.set_page_config(page_title="Optimizador de Liga UNIE", layout="wide")
st.title("Optimización de Audiencias: Temporada Completa")
st.info(
    "En este archivo se realiza el **estudio y optimización de  las 38 jornadas para maximizar la audiencia total de la liga**.\n\n"
    "Si desea realizar el **estudio de una única jornada**, "
    "debe ejecutar el archivo **laliga** disponible en el panel lateral izquierdo, encima de este archivo.\n\n" 
    "Al tratarse de 38 jornadas el algoritmo tarda un poco en calcular múltiples poblaciones por lo que se ha eliminado el gráfico.\n\n Las métricas que aparecen en pantalla se irán actualizando cada ciertas generaciones " \
    "seleccionadas en la barra lateral.\n\n"
    "La mutación consiste en cambiar un partido a un nuevo horario al azar pero si con este cambio la jornada deja de ser válida, entonces no se producirá ese cambio, y esto se hace para cada jornada."
)

# Barra lateral, en ella se elegirán los parámetros del modelo
with st.sidebar:
    st.header("Configuración del Algoritmo")
    gen_max = st.slider("Generaciones Máximas", 50, 1000, 200)
    pob_size = st.slider("Tamaño de Población", 100, 1000, 300)
    mostrar = st.slider("Cada cuantas generaciones le gustaría que se mostraran las métricas", 1, 20, 5)
    mutacion= st.slider("Seleccione el porcentaje que tendrá la mutación de suceder",0, 100, 10)
    
    st.divider()
    
    # Inicialización de estados, tendran nombres distintos respecto a la jornada porque stramlit no diferencia que estado es de que archivo
    #  y los mezclaba dando error de primera aunque una vez le dabas a ejecutar iba sin problemas, pero para evitar que saltase ese error de primera

    if 'ga_running' not in st.session_state: 
        st.session_state.ga_running = False
    if 'gen_actual' not in st.session_state: 
        st.session_state.gen_actual = 0
    if 'historial_audiencias' not in st.session_state: 
        st.session_state.historial_audiencias = []
    if 'mejor_audiencia' not in st.session_state: 
        st.session_state.mejor_audiencia = -1
    if 'mejor_calendario' not in st.session_state: 
        st.session_state.mejor_calendario = None
    if 'mapa_maestro' not in st.session_state: 
        st.session_state.mapa_maestro = None

   # Se crean los botones uno para iniciar y otro para finalizar
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Iniciar Liga"):
            # Al iniciar se cambia el estado de ga_running para que se cumplan las condiciones del bucle
            st.session_state.ga_running = True
            st.session_state.gen_actual = 0
            st.session_state.historial_audiencias = []
            st.session_state.mejor_audiencia = -1
            st.session_state.mejor_calendario = None
            # Generar el calendario base (quién contra quién) una sola vez
            mapa = generar_calendario_base_liga(EQUIPOS)
            random.shuffle(mapa)
            st.session_state.mapa_maestro = mapa
            # Población inicial de matrices 38x10 (jornadas x partidos en cada jornada)
            st.session_state.poblacion = []
            for _ in range(pob_size):
                # Pasamos el mapa maestro y la lista de slots que definiste al principio del script
                individuo = generar_individuo_liga_completa(st.session_state.mapa_maestro, slots_disponibles)
                st.session_state.poblacion.append(individuo)

            st.rerun()

    with col_btn2:
        if st.button("Detener"):
            # Al detener se cambia el estado de ga_running para que no se cumplan las condiciones del bucle
            st.session_state.ga_running = False

# Extraemos los valores actuales del session_state para que no se pierdan al recargar
aud_actual = st.session_state.historial_audiencias[-1] if st.session_state.historial_audiencias else 0.0
gen_actual = st.session_state.gen_actual
mejor_aud = st.session_state.mejor_audiencia if st.session_state.mejor_audiencia > 0 else 0.0

# Para las jornadas válidas, necesitamos calcularlas del mejor individuo 
n_validas = 0
if (
    st.session_state.mejor_calendario is not None
    and st.session_state.mapa_maestro is not None
):
    _,_, n_validas = fitness_liga_completa(
        st.session_state.mejor_calendario,
        st.session_state.mapa_maestro,
        slots_disponibles
    )

# se divide en 4 columnas que mostrarán la audiencia actual, la audiencia record, la generacion actual/gen_max y las jornadas validas/jornadas totales (38)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Audiencia Actual (M)", f"{aud_actual:.2f}")
m2.metric("Generación", f"{gen_actual}/{gen_max}")
m3.metric("Récord (M)", f"{mejor_aud:.2f}")

# Color dinámico para las jornadas válidas
color_delta = "normal" if n_validas < 38 else "inverse"
m4.metric("Jornadas Válidas", f"{int(n_validas)}/38", 
          delta=int(n_validas - 38) if n_validas < 38 else "Completo", 
          delta_color=color_delta)


#  Empieza la iteración GA si ga_running esta en true y no se ha llegado a la última generación
if st.session_state.ga_running and st.session_state.gen_actual < gen_max:
    # Se cambian las métricas cada vez que pase 'mostrar' generaciones o se llegue al final
    pasos = min(mostrar, gen_max - st.session_state.gen_actual)
    poblacion = st.session_state.poblacion
    mapa = st.session_state.mapa_maestro
    
  
    for _ in range(pasos):
        puntuados = []
        for ind in poblacion:
            fit, aud_real, validas = fitness_liga_completa(ind, mapa, slots_disponibles)
            # Guardamos como diccionario para que sea más legible
            puntuados.append({
                "ind": ind, 
                "fit": fit, 
                "aud_real": aud_real, 
                "validas": validas
            })

        # Ordenamos por la variable 'fitness'
        puntuados.sort(key=lambda x: x["fit"], reverse=True)

        # Extraemos el mejor del lote
        mejor_del_lote = puntuados[0] 

        # Actualizamos las métricas
        if mejor_del_lote["aud_real"] > st.session_state.mejor_audiencia:
            st.session_state.mejor_audiencia = mejor_del_lote["aud_real"]
            st.session_state.mejor_calendario = copy.deepcopy(mejor_del_lote["ind"])

        st.session_state.historial_audiencias.append(mejor_del_lote["aud_real"])

        # Se crea la Nueva población pasando los dos primeros y el resto se rellenan 1) realizando la seleccion por torneo de la funcion para selecionar a los padres 2) se cruzan 3) se mutan si se cumplen las condiciones 
        nueva_pob = [
            copy.deepcopy(puntuados[0]["ind"]), 
            copy.deepcopy(puntuados[1]["ind"])
        ]

        while len(nueva_pob) < pob_size:
            p1 = seleccionar_ganador_torneo_liga(poblacion, mapa, slots_disponibles)
            p2 = seleccionar_ganador_torneo_liga(poblacion, mapa, slots_disponibles)
            hijo = cruzar_padres_liga(p1, p2)
            hijo = mutacion_liga(hijo, slots_disponibles,mutacion/100)
            nueva_pob.append(hijo)
            
        poblacion = nueva_pob
        st.session_state.gen_actual += 1

    st.session_state.poblacion = poblacion
    st.rerun() 
# Una vez se finalize el modelo (por el usuario o que llega al máximo de iteraciones) divide la pantalla en tres pestañas (tablas) 
if (
    st.session_state.mejor_calendario is not None
    and st.session_state.mapa_maestro is not None
):
    st.divider()
    tab1, tab2, tab3 = st.tabs(
        ["Calendario por Jornada", "Buscador por Equipo", "Datos de la Temporada"]
    )

    tabla_completa = construir_tabla_liga(
        st.session_state.mejor_calendario,
        st.session_state.mapa_maestro,
        slots_disponibles
    )
    df_completo = pd.DataFrame(tabla_completa)
    # En la tabla 1 habrá un buscador de jornadas, en el que se pueden ir viendo todas los jornadas de una en una
    with tab1:
        j_num = st.select_slider("Selecciona la Jornada", options=range(1, 39))
        df_jornada = df_completo[df_completo["Jornada"] == j_num]
        st.table(df_jornada.drop(columns=["Jornada"]))
    # en la tabbla 2 se puede buscar por equipo, aparecerán todos los partidos de ese equipo
    with tab2:
        equipo_sel = st.selectbox("Selecciona un equipo para ver su calendario", EQUIPOS)
        df_equipo = df_completo[df_completo["Partido"].str.contains(equipo_sel)]
        st.dataframe(df_equipo, use_container_width=True, hide_index=True)
    # en la tabla 3 aparecerán la audiencia fianl y la audiencia media por jornada
    with tab3:
        st.subheader("Estadísticas de Optimización")
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.write("**Audiencia Total Alcanzada:**")
            st.title(f"{st.session_state.mejor_audiencia:.2f} M")
        with col_res2:
            st.write("**Promedio por Jornada:**")
            st.title(f"{(st.session_state.mejor_audiencia/38):.2f} M")