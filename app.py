import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from ortools.sat.python import cp_model
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly.express as px

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================
st.set_page_config(layout="wide", page_title="Optimizador Consolidado - Madera Verde")

st.title("🌲 Optimizador de Carga: Lógica de Grúa (Pares + Deslizamiento)")
st.markdown("""
**Nueva Lógica Operacional:**
1.  **Picking Doble:** El algoritmo simula a la grúa tomando dos paquetes compatibles (mismo ancho, largo similar).
2.  **Ajuste Dinámico (Slide):** El paquete superior puede **deslizarse hasta un 20%** longitudinalmente sobre el inferior para encajar mejor.
3.  **Restricciones:** Carga pegada a muros y separación mínima entre filas.
""")

# ==============================================================================
# 1. FUNCIONES DE AGRUPACIÓN (CLUSTERING VISUAL)
# ==============================================================================
def graficar_clusters(df_input, n_clusters=5):
    """Visualización 3D para entender la mezcla de carga."""
    df_viz = df_input.copy()
    features = df_viz[['Largo', 'Ancho', 'Alto', 'Peso']]
    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_viz['Cluster'] = kmeans.fit_predict(feat_scaled).astype(str)
    
    fig = px.scatter_3d(
        df_viz, x='Largo', y='Peso', z='Ancho', color='Cluster', size='Alto',
        hover_data=['ID', 'Alto', 'Peso'],
        title=f"Agrupación de Inventario ({n_clusters} Clusters)",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig.update_layout(height=500, margin=dict(l=0, r=0, b=0, t=40))
    return fig

# ==============================================================================
# 2. LÓGICA DE "PICKING" (EMPAREJAMIENTO PREVIO)
# ==============================================================================
def generar_pares_logicos(df_items):
    """
    Simula la inteligencia de la grúa antes de entrar al contenedor.
    Agrupa items en pares (Base + Top) si son compatibles.
    """
    # Ordenamos por Ancho (crítico para las uñas), luego Largo, luego Peso
    items = df_items.sort_values(by=['Ancho', 'Largo', 'Peso'], ascending=[False, False, False]).to_dict('records')
    pares = []
    usados = set()
    
    TOLERANCIA_ANCHO = 5   # cm (Para que las uñas agarren bien ambos)
    TOLERANCIA_LARGO = 40  # cm (Diferencia máxima permitida entre arriba y abajo)
    
    for i in range(len(items)):
        if i in usados: continue
            
        item_base = items[i]
        mejor_pareja_idx = -1
        
        # Buscamos compañero
        for j in range(i + 1, len(items)):
            if j in usados: continue
            candidato = items[j]
            
            # Regla 1: Ancho similar
            if abs(candidato['Ancho'] - item_base['Ancho']) > TOLERANCIA_ANCHO:
                continue
            
            # Regla 2: Largo similar
            if abs(candidato['Largo'] - item_base['Largo']) > TOLERANCIA_LARGO:
                continue
            
            mejor_pareja_idx = j
            break
        
        if mejor_pareja_idx != -1:
            # Creamos PAR
            item_top = items[mejor_pareja_idx]
            pares.append({
                'tipo': 'par',
                'base': item_base,
                'top': item_top,
                # Dimensiones nominales del bloque (el solver ajustará el offset)
                'Largo_Ref': max(item_base['Largo'], item_top['Largo']),
                'Ancho_Ref': max(item_base['Ancho'], item_top['Ancho']),
                'Peso_Total': item_base['Peso'] + item_top['Peso']
            })
            usados.add(i)
            usados.add(mejor_pareja_idx)
        else:
            # SINGLE (Va solo en el piso)
            pares.append({
                'tipo': 'single',
                'base': item_base,
                'top': None,
                'Largo_Ref': item_base['Largo'],
                'Ancho_Ref': item_base['Ancho'],
                'Peso_Total': item_base['Peso']
            })
            usados.add(i)
            
    return pares

# ==============================================================================
# 3. MOTOR DE OPTIMIZACIÓN (SOLVER CON OFFSET)
# ==============================================================================
def resolver_contenedor_consolidado(lista_pares, cont_l, cont_w, cont_h, max_peso):
    model = cp_model.CpModel()
    n_units = len(lista_pares)
    
    # --- CONFIGURACIÓN DE HOLGURAS ---
    GAP_Y = 10        
    MARGIN_DOOR = 10  
    
    # --- VARIABLES DE DECISIÓN PRINCIPALES ---
    x = [model.NewIntVar(0, cont_l, f'x_{i}') for i in range(n_units)]
    y = [model.NewIntVar(0, cont_w, f'y_{i}') for i in range(n_units)]
    rotated = [model.NewBoolVar(f'rot_{i}') for i in range(n_units)]
    placed = [model.NewBoolVar(f'placed_{i}') for i in range(n_units)]
    stick_left = [model.NewBoolVar(f'left_{i}') for i in range(n_units)]
    
    # Variables auxiliares
    offset_top = []
    abs_offsets = [] # NUEVO: Para medir cuánto se deslizó (valor absoluto)
    
    l_base_eff = [model.NewIntVar(0, 3000, f'lbe_{i}') for i in range(n_units)]
    w_base_eff = [model.NewIntVar(0, 3000, f'wbe_{i}') for i in range(n_units)]
    total_weight_var = model.NewIntVar(0, max_peso, 'total_weight')
    
    # Variables para CoG
    moments_x = []
    moments_y = []

    # Variables para Bounding Box (Límites reales anticolisión)
    x_start = [model.NewIntVar(0, cont_l, f'xs_{i}') for i in range(n_units)]
    x_end   = [model.NewIntVar(0, cont_l, f'xe_{i}') for i in range(n_units)]
    y_end_abs = [model.NewIntVar(0, cont_w, f'ye_{i}') for i in range(n_units)]

    # --- BUCLE DE CREACIÓN DE VARIABLES Y RESTRICCIONES ---
    for i in range(n_units):
        u = lista_pares[i]
        base = u['base']
        w_b = int(base['Peso'])

        # 1. Geometría Base
        model.Add(l_base_eff[i] == base['Largo']).OnlyEnforceIf(rotated[i].Not())
        model.Add(w_base_eff[i] == base['Ancho']).OnlyEnforceIf(rotated[i].Not())
        model.Add(l_base_eff[i] == base['Ancho']).OnlyEnforceIf(rotated[i])
        model.Add(w_base_eff[i] == base['Largo']).OnlyEnforceIf(rotated[i])
        
        # 2. Estrategia de Muros
        model.Add(y[i] == 0).OnlyEnforceIf([placed[i], stick_left[i]])
        model.Add(y[i] == cont_w - w_base_eff[i]).OnlyEnforceIf([placed[i], stick_left[i].Not()])
        
        # Limpieza de no colocados
        model.Add(x[i] == 0).OnlyEnforceIf(placed[i].Not())
        model.Add(y[i] == 0).OnlyEnforceIf(placed[i].Not())

        # 3. Cálculo de Momentos BASE (X e Y)
        cx_base_2 = model.NewIntVar(0, cont_l * 2, f'cxb2_{i}')
        model.Add(cx_base_2 == x[i] * 2 + l_base_eff[i])
        
        cy_base_2 = model.NewIntVar(0, cont_w * 2, f'cyb2_{i}')
        model.Add(cy_base_2 == y[i] * 2 + w_base_eff[i])

        mx_b = model.NewIntVar(0, cont_l * 2 * w_b, f'mx_b_{i}')
        model.Add(mx_b == cx_base_2 * w_b).OnlyEnforceIf(placed[i])
        model.Add(mx_b == 0).OnlyEnforceIf(placed[i].Not())
        moments_x.append(mx_b)

        my_b = model.NewIntVar(0, cont_w * 2 * w_b, f'my_b_{i}')
        model.Add(my_b == cy_base_2 * w_b).OnlyEnforceIf(placed[i])
        model.Add(my_b == 0).OnlyEnforceIf(placed[i].Not())
        moments_y.append(my_b)

        # 4. Lógica PAR vs SINGLE
        if u['tipo'] == 'par':
            top = u['top']
            w_t = int(top['Peso'])
            
            # --- CAMBIO 1: AUMENTAR LÍMITE AL 80% ---
            max_slide = int(base['Largo'] * 0.20)
            off = model.NewIntVar(-max_slide, max_slide, f'off_{i}')
            offset_top.append(off)

            # --- CAMBIO 2: VARIABLE PARA PENALIZACIÓN (Valor Absoluto) ---
            abs_off = model.NewIntVar(0, max_slide, f'abs_off_{i}')
            model.AddAbsEquality(abs_off, off)
            abs_offsets.append(abs_off)
            
            # Altura
            h_tot = base['Alto'] + top['Alto']
            model.Add(h_tot <= cont_h).OnlyEnforceIf(placed[i])
            
            # Reglas Slide
            model.Add(off == 0).OnlyEnforceIf(rotated[i]) # Si rota, no desliza
            model.Add(x[i] + off >= 0).OnlyEnforceIf([placed[i], rotated[i].Not()])
            model.Add(x[i] + off + top['Largo'] <= cont_l - MARGIN_DOOR).OnlyEnforceIf([placed[i], rotated[i].Not()])

            # Dimensiones Top Efectivas
            l_top_eff = model.NewIntVar(0, 3000, f'lte_{i}')
            w_top_eff = model.NewIntVar(0, 3000, f'wte_{i}')
            model.Add(l_top_eff == top['Largo']).OnlyEnforceIf(rotated[i].Not())
            model.Add(w_top_eff == top['Ancho']).OnlyEnforceIf(rotated[i].Not())
            model.Add(l_top_eff == top['Ancho']).OnlyEnforceIf(rotated[i])
            model.Add(w_top_eff == top['Largo']).OnlyEnforceIf(rotated[i])

            # Momento Top
            cx_top_2 = model.NewIntVar(-max_slide*2, (cont_l + max_slide)*2, f'cxt2_{i}')
            model.Add(cx_top_2 == (x[i] + off) * 2 + l_top_eff)
            
            cy_top_2 = model.NewIntVar(0, cont_w * 2, f'cyt2_{i}')
            model.Add(cy_top_2 == y[i] * 2 + w_top_eff)

            mx_t = model.NewIntVar(-cont_l * 2 * w_t, cont_l * 2 * w_t, f'mx_t_{i}')
            model.Add(mx_t == cx_top_2 * w_t).OnlyEnforceIf(placed[i])
            model.Add(mx_t == 0).OnlyEnforceIf(placed[i].Not())
            moments_x.append(mx_t)

            my_t = model.NewIntVar(0, cont_w * 2 * w_t, f'my_t_{i}')
            model.Add(my_t == cy_top_2 * w_t).OnlyEnforceIf(placed[i])
            model.Add(my_t == 0).OnlyEnforceIf(placed[i].Not())
            moments_y.append(my_t)
            
            # --- BOUNDING BOX (Par) ---
            # x_start = min(x_base, x_top)
            model.AddMinEquality(x_start[i], [x[i], x[i] + off])
            
            # x_end = max(fin_base, fin_top)
            end_base = model.NewIntVar(0, cont_l, f'eb_{i}')
            end_top = model.NewIntVar(-1000, cont_l+1000, f'et_{i}')
            model.Add(end_base == x[i] + l_base_eff[i])
            model.Add(end_top == x[i] + off + l_top_eff)
            model.AddMaxEquality(x_end[i], [end_base, end_top])
            
            # y_end = y + max(w_base, w_top)
            max_w_stack = model.NewIntVar(0, 3000, f'mw_{i}')
            model.AddMaxEquality(max_w_stack, [w_base_eff[i], w_top_eff])
            model.Add(y_end_abs[i] == y[i] + max_w_stack)

        else:
            # Single
            offset_top.append(model.NewConstant(0))
            abs_offsets.append(model.NewConstant(0)) # No hay costo de slide
            
            model.Add(base['Alto'] <= cont_h).OnlyEnforceIf(placed[i])
            
            # Bounding Box Simple
            model.Add(x_start[i] == x[i])
            model.Add(x_end[i] == x[i] + l_base_eff[i])
            model.Add(y_end_abs[i] == y[i] + w_base_eff[i])

    # --- NO SUPERPOSICIÓN (USANDO BOUNDING BOX) ---
    for i in range(n_units):
        for j in range(i + 1, n_units):
            left = model.NewBoolVar(f'{i}_L_{j}')
            right = model.NewBoolVar(f'{i}_R_{j}')
            back = model.NewBoolVar(f'{i}_B_{j}')
            front = model.NewBoolVar(f'{i}_F_{j}')
            
            model.Add(x_end[i] <= x_start[j]).OnlyEnforceIf(left)
            model.Add(x_start[i] >= x_end[j]).OnlyEnforceIf(right)
            model.Add(y_end_abs[i] + GAP_Y <= y[j]).OnlyEnforceIf(back)
            model.Add(y[i] >= y_end_abs[j] + GAP_Y).OnlyEnforceIf(front)
            
            model.AddBoolOr([left, right, back, front]).OnlyEnforceIf([placed[i], placed[j]])
        
        # Limites Contenedor Globales
        model.Add(x_end[i] <= cont_l - MARGIN_DOOR).OnlyEnforceIf(placed[i])
        model.Add(y_end_abs[i] <= cont_w).OnlyEnforceIf(placed[i])

    # --- OBJETIVOS Y RESTRICCIONES DURAS ---
    model.Add(total_weight_var == sum(placed[i] * int(u['Peso_Total']) for i, u in enumerate(lista_pares)))
    model.Add(total_weight_var <= max_peso)
    
    # Restricciones CoG
    LIMIT_X_MIN = (600 - 60) * 2
    LIMIT_X_MAX = (600 + 60) * 2
    LIMIT_Y_MIN = 195 
    LIMIT_Y_MAX = 275 

    sum_mx = sum(moments_x)
    sum_my = sum(moments_y)
    
    model.Add(sum_mx >= LIMIT_X_MIN * total_weight_var)
    model.Add(sum_mx <= LIMIT_X_MAX * total_weight_var)
    model.Add(sum_my >= LIMIT_Y_MIN * total_weight_var)
    model.Add(sum_my <= LIMIT_Y_MAX * total_weight_var)

    # ==========================================================================
    # FUNCIÓN OBJETIVO FINAL
    # ==========================================================================
    # 1. Prioridad Máxima: Maximizar Peso ( * 10,000 )
    # 2. Prioridad Media: Compactar hacia el fondo (- sum(x))
    # 3. Prioridad Ajuste: Minimizar deslizamiento (- sum(abs_offsets))
    #
    # Al restar abs_offsets, el solver intentará mantenerlos en 0 a menos que
    # moverlos sea la única forma de cargar más peso o cumplir el CoG.
    
    model.Maximize(
        total_weight_var * 10000 
        - sum(x) 
        - sum(abs_offsets) * 5  # Factor de penalización (ajustable)
    )

    # --- SOLVE ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 40.0
    solver.parameters.num_workers = 8
    status = solver.Solve(model)

    results = []
    ids_usados = []
    cg_x_final, cg_y_final = 600.0, 117.5

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        peso_final = solver.Value(total_weight_var)
        val_mx = solver.Value(sum_mx)
        val_my = solver.Value(sum_my)
        
        if peso_final > 0:
            cg_x_final = val_mx / (2 * peso_final)
            cg_y_final = val_my / (2 * peso_final)

        for i in range(n_units):
            if solver.Value(placed[i]):
                u = lista_pares[i]
                base = u['base']
                
                xx = solver.Value(x[i])
                yy = solver.Value(y[i])
                rr = solver.Value(rotated[i]) == 1
                
                l_fin = base['Ancho'] if rr else base['Largo']
                w_fin = base['Largo'] if rr else base['Ancho']
                
                results.append({
                    'ID': base['ID'],
                    'x': xx, 'y': yy, 'z': 0,
                    'Largo': l_fin, 'Ancho': w_fin, 'Alto': base['Alto'],
                    'Peso': base['Peso'], 'Color': base['Color'],
                    'Rotado': 'Sí' if rr else 'No', 'Piso': '1 (Base)',
                    'Offset_Ref': 0
                })
                ids_usados.append(base['ID'])
                
                if u['tipo'] == 'par':
                    top = u['top']
                    off_val = solver.Value(offset_top[i])
                    
                    if rr:
                        xt, yt = xx, yy
                        lt, wt = top['Ancho'], top['Largo']
                    else:
                        xt = xx + off_val
                        yt = yy 
                        lt, wt = top['Largo'], top['Ancho']
                    
                    results.append({
                        'ID': top['ID'],
                        'x': xt, 'y': yt, 'z': base['Alto'],
                        'Largo': lt, 'Ancho': wt, 'Alto': top['Alto'],
                        'Peso': top['Peso'], 'Color': top['Color'],
                        'Rotado': 'Sí' if rr else 'No', 'Piso': '2 (Sup)',
                        'Offset_Ref': off_val
                    })
                    ids_usados.append(top['ID'])
        
        return pd.DataFrame(results), peso_final, ids_usados, (cg_x_final, cg_y_final)
    else:
        return pd.DataFrame(), 0, [], (600, 117.5)
    
def ejecutar_optimizacion_flota(df_total, n_contenedores, max_peso):
    contenedores_res = {}
    items_pendientes = df_total.copy()
    
    progreso = st.progress(0)
    status = st.empty()

    for i in range(1, n_contenedores + 1):
        if items_pendientes.empty: break
            
        status.markdown(f"**Contenedor {i}:** Generando pares y optimizando estiba...")
        
        batch_size = 80 
        batch = items_pendientes.head(batch_size)
        
        pares_candidatos = generar_pares_logicos(batch)
        
        # CAMBIO AQUÍ: Desempaquetamos la tupla extra (coords_cg)
        df_cargado, peso, ids, coords_cg = resolver_contenedor_consolidado(
            pares_candidatos, 1200, 235, 269, int(max_peso)
        )
        
        if not df_cargado.empty:
            df_cargado['m3'] = (df_cargado['Largo']*df_cargado['Ancho']*df_cargado['Alto'])/1e6
            
            contenedores_res[f"Contenedor {i}"] = {
                "items": df_cargado,
                "peso_total": peso,
                "m3_total": df_cargado['m3'].sum(),
                "n_bultos": len(df_cargado),
                "cg_x": coords_cg[0],  # Usamos el valor directo
                "cg_y": coords_cg[1]   # Usamos el valor directo
            }
            items_pendientes = items_pendientes[~items_pendientes['ID'].isin(ids)]
        
        progreso.progress(i / n_contenedores)

    status.success("✅ ¡Planificación completada!")
    progreso.empty()
    return contenedores_res, items_pendientes

# ==============================================================================
# 4. INTERFAZ DE USUARIO
# ==============================================================================
with st.sidebar:
    st.header("1. Datos de Entrada")
    uploaded_file = st.file_uploader("Excel (ID, Largo, Ancho, Alto, Peso)", type=["xlsx"])
    
    st.header("2. Parámetros")
    n_cont = st.number_input("N° Contenedores", 1, 50, 5)
    max_w = st.number_input("Peso Máx (kg)", value=26000)
    
    btn_calc = st.button("🚀 Calcular Carga", type="primary")

if uploaded_file and btn_calc:
    try:
        df_input = pd.read_excel(uploaded_file)
        # Limpieza y generación de colores
        cols_num = ['Largo', 'Ancho', 'Alto', 'Peso']
        for c in cols_num: df_input[c] = pd.to_numeric(df_input[c], errors='coerce').fillna(0).astype(int)
        
        if 'Color' not in df_input.columns:
            df_input['Color'] = [f'rgb({random.randint(100,200)},{random.randint(80,150)},{random.randint(50,120)})' for _ in range(len(df_input))]

        st.session_state['res'] = ejecutar_optimizacion_flota(df_input, n_cont, max_w)
    except Exception as e:
        st.error(f"Error procesando archivo: {e}")

# ==============================================================================
# 5. VISUALIZACIÓN DE RESULTADOS (COMPATIBLE VERSIONES ANTIGUAS)
# ==============================================================================
if 'res' in st.session_state:
    resultados, sobrante = st.session_state['res']
    
    # Placeholder para limpiar la pantalla
    viz_placeholder = st.empty()
    
    if resultados:
        with viz_placeholder.container():
            # ---------------- SELECTOR ----------------
            col_sel, _ = st.columns([1, 3])
            with col_sel:
                cont_sel = st.radio("Seleccionar Contenedor:", list(resultados.keys()))
            
            data_c = resultados[cont_sel]
            df_items = data_c['items']

            # --- DATOS ---
            cg_real_x = data_c.get('cg_x', 600)
            cg_real_y = data_c.get('cg_y', 117.5)
            cg_ideal_x = 1200 / 2  
            cg_ideal_y = 235 / 2   

            vol_max_m3 = (1200 * 235 * 269) / 1_000_000
            vol_ocupado = data_c['m3_total']
            pct_vol = (vol_ocupado / vol_max_m3) * 100
            peso_ocupado = data_c['peso_total']
            pct_peso = (peso_ocupado / max_w) * 100
            desv_x = cg_real_x - cg_ideal_x

            # ---------------- MÉTRICAS DINÁMICAS ----------------
            # TRUCO: Agregamos 'cont_sel' al título. 
            # Si el título cambia, Streamlit OBLIGA a redibujar el número.
            
            st.markdown(f"### 📊 Reporte: {cont_sel}")
            
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1: 
                # Eliminamos 'key=' y ponemos el nombre del contenedor en el Label
                st.metric(f"⚖️ Peso Total [{cont_sel}]", f"{peso_ocupado:,.0f} kg", f"{pct_peso:.1f}% Cap.")
            with col_m2: 
                st.metric(f"📦 Volumen [{cont_sel}]", f"{vol_ocupado:.2f} m³", f"{pct_vol:.1f}% Ocup.")
            with col_m3:
                st.metric(f"🎯 Desviación Largo [{cont_sel}]", f"{desv_x:.0f} cm", delta_color="off")

            st.progress(pct_peso / 100, text=f"Uso de Peso Permitido: {pct_peso:.1f}%")

            # ---------------- VALIDACIÓN ----------------
            cumple_largo = abs(desv_x) <= 60
            cumple_ancho = abs(cg_real_y - cg_ideal_y) <= 20
            
            if cumple_largo and cumple_ancho:
                st.success(f"✅ **APROBADO** | Distribución segura.")
            else:
                errores = []
                if not cumple_largo:
                    dir_error = "hacia la PUERTA" if cg_real_x > cg_ideal_x else "hacia el FONDO"
                    errores.append(f"⚠️ **Desbalance Longitudinal:** Carga desplazada {dir_error}.")
                if not cumple_ancho:
                    dir_error = "a la IZQUIERDA" if cg_real_y < cg_ideal_y else "a la DERECHA"
                    errores.append(f"⚠️ **Desbalance Transversal:** Carga inclinada {dir_error}.")
                st.error(" | ".join(errores))

            # ---------------- TABS ----------------
            tabs = st.tabs(["Vista 3D Interactiva", "Planos 2D (Planta)", "Reporte Detallado"])

            # TAB 1: 3D
            with tabs[0]: 
                fig = go.Figure()
                # Items
                for _, r in df_items.iterrows():
                    l, w, h = r['Largo'], r['Ancho'], r['Alto']
                    xe = [r['x'], r['x']+l, r['x']+l, r['x'], r['x'], r['x']+l, r['x']+l, r['x']]
                    ye = [r['y'], r['y'], r['y']+w, r['y']+w, r['y'], r['y'], r['y']+w, r['y']+w]
                    ze = [r['z'], r['z'], r['z'], r['z'], r['z']+h, r['z']+h, r['z']+h, r['z']+h]
                    
                    fig.add_trace(go.Mesh3d(
                        x=xe, y=ye, z=ze, 
                        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                        color=r['Color'], opacity=1, name=str(r['ID']), hoverinfo='text',
                        text=f"ID: {r['ID']}<br>Peso: {r['Peso']} kg"
                    ))
                
                # Puntos CoG
                fig.add_trace(go.Scatter3d(
                    x=[cg_ideal_x], y=[cg_ideal_y], z=[135],
                    mode='markers', marker=dict(size=8, color='green', symbol='circle'),
                    name='Centro Ideal'
                ))
                fig.add_trace(go.Scatter3d(
                    x=[cg_real_x], y=[cg_real_y], z=[135],
                    mode='markers', marker=dict(size=10, color='red', symbol='diamond'),
                    name='Centro Real'
                ))

                # Wireframe
                CL, CW, CH = 1200, 235, 269
                xc = [0, CL, CL, 0, 0, 0, CL, CL, 0, 0, CL, CL, CL, CL, 0, 0]
                yc = [0, 0, CW, CW, 0, 0, 0, CW, CW, 0, 0, 0, CW, CW, CW, CW]
                zc = [0, 0, 0, 0, 0, CH, CH, CH, CH, CH, CH, 0, 0, CH, CH, 0]
                fig.add_trace(go.Scatter3d(x=xc, y=yc, z=zc, mode='lines', line=dict(color='black', width=3), hoverinfo='none'))
                
                fig.update_layout(scene=dict(aspectmode='data'), height=600, margin=dict(l=0,r=0,b=0,t=0))
                
                # Mantenemos key aleatoria en gráficas porque st.plotly_chart SI soporta keys en versiones viejas
                st.plotly_chart(fig, use_container_width=True, key=f"3d_{cont_sel}_{random.randint(0,10000)}")

        # ----------------------------------------------------------------------
        # TAB 2: PLANOS 2D CON HOVER DETALLADO
        # ----------------------------------------------------------------------
        with tabs[1]: 
            c1, c2 = st.columns(2)
            
            def plot_2d_interactivo(df_sub, title):
                f2 = go.Figure()
                
                # 1. Dibujar Contorno Contenedor
                f2.add_shape(type="rect", x0=0, y0=0, x1=1200, y1=235, line=dict(color="black", width=3))
                
                # 2. Dibujar Paquetes
                for _, r in df_sub.iterrows():
                    # Rectángulo visual
                    f2.add_shape(
                        type="rect", 
                        x0=r['x'], y0=r['y'], 
                        x1=r['x']+r['Largo'], y1=r['y']+r['Ancho'], 
                        fillcolor=r['Color'], 
                        line=dict(color='black', width=1),
                        opacity=1
                    )
                    
                    # 3. Texto Invisible (o ID) que lleva el HOVER
                    # Construimos el HTML para el tooltip
                    tooltip_html = (
                        f"<b>ID: {r['ID']}</b><br>"
                        f"📏 {r['Largo']} x {r['Ancho']} x {r['Alto']} cm<br>"
                        f"⚖️ {r['Peso']} kg<br>"
                        f"📍 Pos X: {r['x']} | Y: {r['y']}"
                    )
                    
                    # Usamos Scatter text para mostrar el ID y activar el hover
                    # Calculamos contraste de texto (blanco sobre oscuro, negro sobre claro)
                    text_color = 'white' # Default simple
                    
                    f2.add_trace(go.Scatter(
                        x=[r['x'] + r['Largo']/2],
                        y=[r['y'] + r['Ancho']/2],
                        text=str(r['ID']),
                        mode='text',
                        hoverinfo='text',       # Activamos info personalizada
                        hovertext=tooltip_html, # Aquí va el HTML con medidas y peso
                        showlegend=False,
                        textfont=dict(color=text_color, size=11, family="Arial Black")
                    ))
                    
                f2.update_layout(
                    title=dict(text=title, x=0.5),
                    height=400,
                    xaxis=dict(
                        range=[-20, 1220], 
                        title="Largo (cm)",
                        showgrid=True,
                        zeroline=False
                    ),
                    yaxis=dict(
                        range=[-20, 255], 
                        title="Ancho (cm)",
                        scaleanchor="x", 
                        scaleratio=1,
                        showgrid=True,
                        zeroline=False
                    ),
                    margin=dict(l=20, r=20, t=40, b=20),
                    dragmode='pan' # Herramienta de mano por defecto
                )
                return f2

            with c1: 
                st.plotly_chart(
                    plot_2d_interactivo(df_items[df_items['z']==0], "⬇️ Piso 1 (Base)"), 
                    use_container_width=True
                )
            
            with c2: 
                # Solo mostramos piso 2 si hay algo
                df_p2 = df_items[df_items['z']>0]
                if not df_p2.empty:
                    st.plotly_chart(
                        plot_2d_interactivo(df_p2, "⬆️ Piso 2 (Superior)"), 
                        use_container_width=True
                    )
                else:
                    st.info("Este contenedor no tiene carga en el segundo piso.")

            # TAB 3: DATA
            with tabs[2]: st.dataframe(df_items)

    if not sobrante.empty:
        st.error(f"⚠️ Quedaron {len(sobrante)} bultos sin cargar.")