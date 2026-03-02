import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from ortools.sat.python import cp_model
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly.express as px
import itertools
import io
# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================
st.set_page_config(layout="wide", page_title="Optimizador Consolidado - Madera Verde")

st.title("Optimizador de Estiba")
st.markdown("""
**Nueva Lógica Operacional:**
1.  **Tres pisos de carga**
2.  **Slide de 0.05%**
3.  **Considera centro de gravedad**
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
def limpiar_y_preparar_datos(df_raw):
    """Normaliza columnas: Paquete->ID, Largo fmt, Ancho/Alto fijos, Key Agrupación."""
    df = df_raw.copy()
    
    # Función unificada para pasar cualquier medida de Metros a Centímetros
    def metros_a_cm(val):
        try:
            val_str = str(val).replace(',', '.')
            return int(float(val_str) * 100)
        except:
            return 0

    # 1. Formatear Largo 
    df['Largo_cm'] = df['Largo'].apply(metros_a_cm)
    
    # 2. Dimensiones
    # Si quieres que también lea el ancho real del excel en vez de 110, cámbialo a: 
    if 'Ancho' in df.columns:
        df['Ancho_cm'] = df['Ancho'].apply(metros_a_cm) # <--- Ahora SÍ existe la función
    else:
        # Valor de respaldo por si suben un archivo antiguo sin la columna
        df['Ancho_cm'] = 70
    
    if 'Alto' in df.columns:
        df['Alto_cm'] = df['Alto'].apply(metros_a_cm) # <--- Ahora SÍ existe la función
    else:
        # Valor de respaldo por si suben un archivo antiguo sin la columna
        df['Alto_cm'] = 120
    
    # 3. Identificadores y Key de Agrupación (Pedido + Pos)
    df['ID'] = df['Paquete']
    df['Pedido_Key'] = df['Pedido'].astype(str) + "_" + df['Pos Pedido'].astype(str)
    
    # 4. Asegurar Peso numérico
    df['Peso'] = pd.to_numeric(df['Peso'], errors='coerce').fillna(0)
    
    # NUEVO: Leer la columna Volumen del Excel
    if 'Volumen' in df.columns:
        df['Volumen'] = pd.to_numeric(df['Volumen'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
    else:
        df['Volumen'] = 0.0 # Respaldo por si el archivo no la trae
        
    # 5. Generar Color por Pedido para diferenciación visual
    unique_lengths = df['Largo_cm'].unique()
    color_map = {
        l: f'rgb({random.randint(50,220)},{random.randint(50,220)},{random.randint(50,220)})' 
        for l in unique_lengths
    }
    df['Color'] = df['Largo_cm'].map(color_map)

    # 6. Seleccionar y renombrar para el algoritmo
    df_final = df[[
        'ID', 'Largo_cm', 'Ancho_cm', 'Alto_cm', 'Peso', 'Volumen', 'Color', 'Pedido_Key', 'Pedido', 'Pos Pedido'
    ]].rename(columns={'Largo_cm': 'Largo', 'Ancho_cm': 'Ancho', 'Alto_cm': 'Alto'})
    
    # Ordenar para priorizar carga ordenada
    return df_final.sort_values(by=['Pedido_Key', 'Largo'], ascending=[True, False])

def buscar_combinacion_nivel(candidatos, largo_base, max_h_disp, tolerancia=40, max_piezas=4):
    """
    Busca la mejor combinación de 1 a 4 piezas que sumen un largo similar a la base.
    """
    mejor_combo = None
    mejor_diff = float('inf')
    
    # Probamos combinaciones de tamaño 1, 2, 3 y 4
    for r in range(1, max_piezas + 1):
        for combo in itertools.combinations(candidatos, r):
            # Calculamos dimensiones del Macro-Paquete
            largo_total = sum(it['Largo'] for it, idx in combo)
            alto_max = max(it['Alto'] for it, idx in combo)
            
            # Verificamos restricciones
            diff_largo = abs(largo_total - largo_base)
            if diff_largo <= tolerancia and alto_max <= max_h_disp:
                # Nos quedamos con la combinación que más se acerque al largo de la base
                if diff_largo < mejor_diff:
                    mejor_diff = diff_largo
                    mejor_combo = combo
                    
    return mejor_combo

def generar_stacks_logicos(df_items, max_h_cont=269):
    MARGIN_ROOF = 21
    h_max_efectiva = max_h_cont - MARGIN_ROOF
    items = df_items.sort_values(by=['Pedido_Key', 'Largo', 'Peso'], ascending=[True, False, False]).to_dict('records')
    stacks = []
    usados = set()
    
    TOLERANCIA_LARGO = 40  
    
    for i in range(len(items)):
        if i in usados: continue
        
        base = items[i]
        
        # El nivel 1 (base) lo mantenemos como 1 sola pieza principal por estabilidad
        # Formato de Macro-Paquete: {'items_internos': [...], 'Largo': X, 'Ancho': Y, 'Alto': Z, 'Peso': W}
        stack_levels = [{
            'items_internos': [base],
            'Largo': base['Largo'],
            'Ancho': base['Ancho'],
            'Alto': base['Alto'],
            'Peso': base['Peso']
        }]
        
        current_h = base['Alto']
        usados.add(i)
        
        # Intentamos buscar hasta 2 niveles más (pisos 2 y 3)
        for nivel in range(2): 
            h_disponible = h_max_efectiva - current_h
            if h_disponible < 10: break # Ya no hay altura útil
            
            # Filtramos candidatos del mismo pedido y que no superen la altura disponible
            candidatos_validos = [
                (items[j], j) for j in range(i + 1, len(items)) 
                if j not in usados 
                and items[j]['Pedido_Key'] == base['Pedido_Key']
                and items[j]['Alto'] <= h_disponible
            ]
            
            if not candidatos_validos: break
            
            # Llamamos a nuestra nueva lógica combinatoria
            combo_encontrado = buscar_combinacion_nivel(
                candidatos_validos, 
                largo_base=base['Largo'], 
                max_h_disp=h_disponible,
                tolerancia=TOLERANCIA_LARGO,
                max_piezas=4
            )
            
            if combo_encontrado:
                items_combo = [it for it, idx in combo_encontrado]
                indices_combo = [idx for it, idx in combo_encontrado]
                
                # Creamos el Nivel Compuesto (Macro-Paquete)
                nivel_compuesto = {
                    'items_internos': items_combo,
                    'Largo': sum(it['Largo'] for it in items_combo),
                    'Ancho': max(it['Ancho'] for it in items_combo),
                    'Alto': max(it['Alto'] for it in items_combo),
                    'Peso': sum(it['Peso'] for it in items_combo)
                }
                
                stack_levels.append(nivel_compuesto)
                current_h += nivel_compuesto['Alto']
                usados.update(indices_combo) # Marcamos todas las piezas elegidas como usadas
            else:
                break # Si no pudimos armar un nivel entero, cortamos la torre aquí
        
        # Construimos el metadata del Stack para el Solver
        peso_total = sum(lvl['Peso'] for lvl in stack_levels)
        max_l = max(lvl['Largo'] for lvl in stack_levels)
        max_w = max(lvl['Ancho'] for lvl in stack_levels)
        
        stacks.append({
            'niveles': stack_levels, # <-- AHORA ENVIAMOS NIVELES EN LUGAR DE ITEMS
            'Largo_Ref': max_l,
            'Ancho_Ref': max_w,
            'Peso_Total': peso_total,
            'Pedido_Key': base['Pedido_Key']
        })
            
    return stacks

def resolver_contenedor_consolidado(lista_stacks, cont_l, cont_w, cont_h, max_peso):
    model = cp_model.CpModel()
    n_stacks = len(lista_stacks)
    
    # --- CONFIGURACIÓN ---
    GAP_Y = 11        
    MARGIN_DOOR = 10  
    MARGIN_ROOF = 21
    # --- VARIABLES GLOBALES ---
    block_start = model.NewIntVar(0, cont_l, 'block_start')
    total_weight_var = model.NewIntVar(0, max_peso, 'total_weight')

    # Arrays de variables por Stack
    x_rel = []     # Posición relativa al bloque
    x = []         # Posición absoluta (x_rel + block_start)
    y = []
    rotated = []
    placed = []

    
    offset_upper = [] # Deslizamiento de los pisos superiores respecto a la base
    abs_offsets = [] 
    
    x_start = [] # Bounding Box Real
    x_end = []
    y_end_abs = []
    
    moments_x = []
    moments_y = []

    # --- BUCLE DE GENERACIÓN DE VARIABLES POR STACK ---
    for i in range(n_stacks):
        stk = lista_stacks[i]
        base_item = stk['niveles'][0] # El item base define muchas cosas
        n_pisos = len(stk['niveles'])
        
        # 1. Variables de Decisión
        p_i = model.NewBoolVar(f'placed_{i}')
        placed.append(p_i)
        
        r_i = model.NewBoolVar(f'rot_{i}')
        rotated.append(r_i)
        
        xr_i = model.NewIntVar(0, cont_l, f'x_rel_{i}')
        x_rel.append(xr_i)
        
        x_i = model.NewIntVar(0, cont_l, f'x_{i}')
        x.append(x_i)
        
        y_i = model.NewIntVar(0, cont_w, f'y_{i}')
        y.append(y_i)
            
        # Conexión Bloque
        model.Add(x_i == block_start + xr_i).OnlyEnforceIf(p_i)
        
        # Limpieza si no placed
        model.Add(x_i == 0).OnlyEnforceIf(p_i.Not())
        model.Add(xr_i == 0).OnlyEnforceIf(p_i.Not())
        model.Add(y_i == 0).OnlyEnforceIf(p_i.Not())

        # 2. Dimensiones Efectivas de la BASE
        l_base_eff = model.NewIntVar(0, 3000, f'lbe_{i}')
        w_base_eff = model.NewIntVar(0, 3000, f'wbe_{i}')
        
        model.Add(l_base_eff == base_item['Largo']).OnlyEnforceIf(r_i.Not())
        model.Add(w_base_eff == base_item['Ancho']).OnlyEnforceIf(r_i.Not())
        model.Add(l_base_eff == base_item['Ancho']).OnlyEnforceIf(r_i)
        model.Add(w_base_eff == base_item['Largo']).OnlyEnforceIf(r_i)

        # Definimos 3 posibles estados de ubicación transversal
        y_left = model.NewBoolVar(f'yl_{i}')
        y_right = model.NewBoolVar(f'yr_{i}')
        y_mid = model.NewBoolVar(f'ym_{i}')

        model.AddExactlyOne([y_left, y_right, y_mid])
        # Muros (Y)
        model.Add(y_i == 0).OnlyEnforceIf([p_i, y_left])
        model.Add(y_i == cont_w - w_base_eff).OnlyEnforceIf([p_i, y_right])

        # 3. Lógica de Slide (Solo si hay más de 1 piso)
        if n_pisos > 1:
            max_slide = int(base_item['Largo'] * 0.05)
            off = model.NewIntVar(-max_slide, max_slide, f'off_{i}')
            offset_upper.append(off)
            
            a_off = model.NewIntVar(0, max_slide, f'aoff_{i}')
            model.AddAbsEquality(a_off, off)
            abs_offsets.append(a_off)
            
            # Restricciones Slide
            model.Add(off == 0).OnlyEnforceIf(r_i) # Si rota, no desliza
            model.Add(x_i + off >= 0).OnlyEnforceIf([p_i, r_i.Not()])
            # El límite superior del contenedor se verifica con el Bounding Box más abajo
        else:
            offset_upper.append(model.NewConstant(0))
            abs_offsets.append(model.NewConstant(0))

        # 4. Cálculo de Momentos y Bounding Box Global del Stack
        # Iteramos sobre los items del stack para sumar sus momentos y hallar el contorno
        
        current_stack_moments_x = []
        current_stack_moments_y = []
        
        # Variables para calcular el Bounding Box (BB) del stack completo
        bb_xs = model.NewIntVar(0, cont_l, f'bb_xs_{i}') # X Start Stack
        bb_xe = model.NewIntVar(0, cont_l, f'bb_xe_{i}') # X End Stack
        bb_ye = model.NewIntVar(0, cont_w, f'bb_ye_{i}') # Y End Stack (Ancho máx)
        
        list_ends_x = []
        list_starts_x = []
        list_widths = []

        for idx_it, item in enumerate(stk['niveles']):
            w_kg = int(item['Peso'])
            
            # Dimensiones locales
            l_it_eff = model.NewIntVar(0, 3000, f'lie_{i}_{idx_it}')
            w_it_eff = model.NewIntVar(0, 3000, f'wie_{i}_{idx_it}')
            
            model.Add(l_it_eff == item['Largo']).OnlyEnforceIf(r_i.Not())
            model.Add(w_it_eff == item['Ancho']).OnlyEnforceIf(r_i.Not())
            model.Add(l_it_eff == item['Ancho']).OnlyEnforceIf(r_i)
            model.Add(w_it_eff == item['Largo']).OnlyEnforceIf(r_i)
            
            # Posición X del item (Base es x_i, Superiores son x_i + off)
            pos_x_it = model.NewIntVar(-1000, 4000, f'px_{i}_{idx_it}')
            if idx_it == 0:
                model.Add(pos_x_it == x_i)
            else:
                model.Add(pos_x_it == x_i + offset_upper[i])
            
            # Momentos
            cx_2 = model.NewIntVar(-2000, 8000, f'cx2_{i}_{idx_it}')
            model.Add(cx_2 == pos_x_it * 2 + l_it_eff)
            
            cy_2 = model.NewIntVar(0, cont_w * 2, f'cy2_{i}_{idx_it}')
            model.Add(cy_2 == y_i * 2 + w_it_eff)
            
            mx = model.NewIntVar(-cont_l*2*w_kg, cont_l*2*w_kg, f'mx_{i}_{idx_it}')
            my = model.NewIntVar(0, cont_w*2*w_kg, f'my_{i}_{idx_it}')
            
            model.Add(mx == cx_2 * w_kg).OnlyEnforceIf(p_i)
            model.Add(mx == 0).OnlyEnforceIf(p_i.Not())
            
            model.Add(my == cy_2 * w_kg).OnlyEnforceIf(p_i)
            model.Add(my == 0).OnlyEnforceIf(p_i.Not())
            
            current_stack_moments_x.append(mx)
            current_stack_moments_y.append(my)
            
            # Para Bounding Box
            start_x_var = model.NewIntVar(0, cont_l, f'sx_{i}_{idx_it}')
            end_x_var = model.NewIntVar(-1000, 4000, f'ex_{i}_{idx_it}')
            
            # Clamp manual para start_x (evitar negativos en BB logic)
            # Como pos_x_it puede ser negativo por el slide, el BB start real es max(0, pos)
            # Pero para colisiones usamos el valor crudo, el contenedor limita el rango.
            model.Add(start_x_var == pos_x_it) 
            model.Add(end_x_var == pos_x_it + l_it_eff)
            
            list_starts_x.append(start_x_var)
            list_ends_x.append(end_x_var)
            list_widths.append(w_it_eff)

        # Definir BB del Stack Completo
        model.AddMinEquality(bb_xs, list_starts_x)
        model.AddMaxEquality(bb_xe, list_ends_x)
        
        max_width_stack = model.NewIntVar(0, 3000, f'mws_{i}')
        model.AddMaxEquality(max_width_stack, list_widths)
        model.Add(bb_ye == y_i + max_width_stack)
        
        # Guardamos para colisiones
        x_start.append(bb_xs)
        x_end.append(bb_xe)
        y_end_abs.append(bb_ye)
        
        # Sumar momentos al global
        moments_x.extend(current_stack_moments_x)
        moments_y.extend(current_stack_moments_y)
        
        # Altura Total
        h_total_stack = sum(it['Alto'] for it in stk['niveles'])
        model.Add(h_total_stack <= cont_h - MARGIN_ROOF).OnlyEnforceIf(p_i)

    # --- COLISIONES ENTRE STACKS ---
    for i in range(n_stacks):
        for j in range(i + 1, n_stacks):
            left = model.NewBoolVar(f'{i}_L_{j}')
            right = model.NewBoolVar(f'{i}_R_{j}')
            back = model.NewBoolVar(f'{i}_B_{j}')
            front = model.NewBoolVar(f'{i}_F_{j}')
            
            model.Add(x_end[i] <= x_start[j]).OnlyEnforceIf(left)
            model.Add(x_start[i] >= x_end[j]).OnlyEnforceIf(right)
            model.Add(y_end_abs[i] + GAP_Y <= y[j]).OnlyEnforceIf(back)
            model.Add(y[i] >= y_end_abs[j] + GAP_Y).OnlyEnforceIf(front)
            
            model.AddBoolOr([left, right, back, front]).OnlyEnforceIf([placed[i], placed[j]])
        
        # Límites globales contenedor
        model.Add(x_end[i] <= cont_l - MARGIN_DOOR).OnlyEnforceIf(placed[i])
        model.Add(y_end_abs[i] <= cont_w).OnlyEnforceIf(placed[i])

    # --- OBJETIVOS DE PESO Y COG ---
    model.Add(total_weight_var == sum(placed[i] * int(stk['Peso_Total']) for i, stk in enumerate(lista_stacks)))
    model.Add(total_weight_var <= max_peso)
    
    # Límites CoG
    LIMIT_X_MIN = (600 - 60) * 2
    LIMIT_X_MAX = (600 + 60) * 2
    LIMIT_Y_MIN = 185 
    LIMIT_Y_MAX = 285 

    sum_mx = sum(moments_x)
    sum_my = sum(moments_y)
    
    model.Add(sum_mx >= LIMIT_X_MIN * total_weight_var)
    model.Add(sum_mx <= LIMIT_X_MAX * total_weight_var)
    model.Add(sum_my >= LIMIT_Y_MIN * total_weight_var)
    model.Add(sum_my <= LIMIT_Y_MAX * total_weight_var)

    # Desviación Ideal X
    ideal_mx = model.NewIntVar(0, cont_l * 2 * max_peso, 'imx')
    model.Add(ideal_mx == total_weight_var * 1200)
    diff_mx = model.NewIntVar(-cont_l*2*max_peso, cont_l*2*max_peso, 'dmx')
    model.Add(diff_mx == sum_mx - ideal_mx)
    abs_diff_mx = model.NewIntVar(0, cont_l*2*max_peso, 'admx')
    model.AddAbsEquality(abs_diff_mx, diff_mx)
    
    # Desviación Ideal Y
    ideal_my = model.NewIntVar(0, cont_w * 2 * max_peso, 'imy')
    model.Add(ideal_my == total_weight_var * 235)
    diff_my = model.NewIntVar(-cont_w*2*max_peso, cont_w*2*max_peso, 'dmy')
    model.Add(diff_my == sum_my - ideal_my)
    abs_diff_my = model.NewIntVar(0, cont_w*2*max_peso, 'admy')
    model.AddAbsEquality(abs_diff_my, diff_my)

    # --- FUNCIÓN OBJETIVO ---
    model.Maximize(
        total_weight_var * 1000000 
        - sum(x_rel) * 10       
        - abs_diff_mx           
        - abs_diff_my
        - sum(abs_offsets) * 5
    )

    # --- SOLVE ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 45.0
    solver.parameters.num_workers = 8
    status = solver.Solve(model)

    results = []
    ids_usados = []
    cg_x_final, cg_y_final = 600.0, 117.5

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        peso_final = solver.Value(total_weight_var)
        val_mx = solver.Value(sum_mx)
        val_my = solver.Value(sum_my)
        inicio_bloque = solver.Value(block_start)
        
        if peso_final > 0:
            cg_x_final = val_mx / (2 * peso_final)
            cg_y_final = val_my / (2 * peso_final)

        for i in range(n_stacks):
            if solver.Value(placed[i]):
                stk = lista_stacks[i]
                
                xx = solver.Value(x[i])
                yy = solver.Value(y[i])
                rr = solver.Value(rotated[i]) == 1
                off_val = solver.Value(offset_upper[i]) if len(stk['niveles']) > 1 else 0
                
                current_z = 0
                
                for idx_lvl, nivel in enumerate(stk['niveles']):
                    # X inicial de este nivel (Base o Superior con slide 5%)
                    inicio_x_nivel = xx if idx_lvl == 0 else xx + off_val
                    offset_interno_x = 0 # Para ubicar piezas consecutivas
                    
                    for item_real in nivel['items_internos']:
                        # Definimos dimensiones finales según rotación general de la torre
                        l_fin = item_real['Ancho'] if rr else item_real['Largo']
                        w_fin = item_real['Largo'] if rr else item_real['Ancho']
                        
                        # Si la torre rotó, las piezas se apilan a lo largo del Eje Y
                        # Si no rotó, se apilan a lo largo del Eje X
                        if rr:
                            final_x = inicio_x_nivel
                            final_y = yy + offset_interno_x
                            offset_interno_x += w_fin # Ocupamos Y
                        else:
                            final_x = inicio_x_nivel + offset_interno_x
                            final_y = yy
                            offset_interno_x += l_fin # Ocupamos X
                    
                        results.append({
                            'ID': item_real['ID'],
                            'x': final_x, 'y': final_y, 'z': current_z,
                            'Largo': l_fin, 'Ancho': w_fin, 'Alto': item_real['Alto'],
                            'Pedido': item_real.get('Pedido', ''),
                            'Pos Pedido': item_real.get('Pos Pedido', ''),
                            'Peso': item_real['Peso'], 'Color': item_real['Color'],
                            'Volumen': item_real.get('Volumen', 0),
                            'Rotado': 'Sí' if rr else 'No', 
                            'Piso': f'Piso {idx_lvl + 1}',
                            'Offset_Ref': off_val if idx_lvl > 0 else 0,
                            'Block_Start': inicio_bloque
                        })
                        ids_usados.append(item_real['ID'])
                    current_z += nivel['Alto']
        
        return pd.DataFrame(results), peso_final, ids_usados, (cg_x_final, cg_y_final)
    else:
        return pd.DataFrame(), 0, [], (600, 117.5)
    
def ejecutar_optimizacion_flota(df_total, max_peso, min_vol=10.0, num_contenedores_fijos=0):
    """
    Optimiza la carga usando los contenedores necesarios o una cantidad fija.
    """
    contenedores_res = {}
    ids_cargados_total = set()
    progreso = st.progress(0)
    status = st.empty()
    
    # ---------------------------------------------------------
    # MODO 1: CANTIDAD FIJA DE CONTENEDORES (Distribución Equitativa)
    # ---------------------------------------------------------
    if num_contenedores_fijos > 0:
        status.markdown(f"**Distribuyendo carga equitativamente en {num_contenedores_fijos} contenedores...**")
        
        # Ordenamos todos los items por Largo y Peso para repartir los más grandes/pesados primero
        df_sorted = df_total.sort_values(by=['Largo', 'Peso'], ascending=[False, False]).reset_index(drop=True)
        
        for cont_idx in range(num_contenedores_fijos):
            # Repartimos estilo "repartir cartas" (Round-Robin) usando el módulo del índice
            items_subset = df_sorted[df_sorted.index % num_contenedores_fijos == cont_idx]
            
            if items_subset.empty:
                continue
                
            status.markdown(f"**Optimizando Contenedor {cont_idx + 1} de {num_contenedores_fijos}...**")
            
            # Tomamos un máximo de 80 items por contenedor para no saturar el solver
            batch = items_subset.head(80) 
            pares_candidatos = generar_stacks_logicos(batch)
            
            df_cargado, peso, ids, coords_cg = resolver_contenedor_consolidado(
                pares_candidatos, 1200, 235, 269, int(max_peso)
            )
            
            if not df_cargado.empty:
                # Usar columna 'Volumen' o calcular matemáticamente
                if 'Volumen' in df_cargado.columns:
                    df_cargado['m3'] = df_cargado['Volumen']
                else:
                    df_cargado['m3'] = (df_cargado['Largo']*df_cargado['Ancho']*df_cargado['Alto'])/1e6
                
                vol_total = df_cargado['m3'].sum()
                
                # RESTRICCIÓN DE VOLUMEN MÍNIMO
                if vol_total >= min_vol:
                    contenedores_res[f"Contenedor {cont_idx + 1}"] = {
                        "items": df_cargado,
                        "peso_total": peso,
                        "m3_total": vol_total,
                        "cg_x": coords_cg[0],
                        "cg_y": coords_cg[1],
                        "pedidos": items_subset['Pedido_Key'].unique().tolist()
                    }
                    ids_cargados_total.update(ids)
            
            progreso.progress((cont_idx + 1) / num_contenedores_fijos)

    # ---------------------------------------------------------
    # MODO 2: AUTOMÁTICO (Agrupado por pedido y llenado al máximo)
    # ---------------------------------------------------------
    else:
        grupos_pedidos = df_total.groupby('Pedido_Key')
        cont_global_idx = 1
        total_grupos = len(grupos_pedidos)
        grupo_actual_idx = 0

        for pedido_key, df_grupo in grupos_pedidos:
            grupo_actual_idx += 1
            status.markdown(f"**Procesando:** {pedido_key}...")
            
            items_pendientes_pedido = df_grupo.copy()
            
            while not items_pendientes_pedido.empty:
                batch = items_pendientes_pedido.head(80)
                pares_candidatos = generar_stacks_logicos(batch)
                
                df_cargado, peso, ids, coords_cg = resolver_contenedor_consolidado(
                    pares_candidatos, 1200, 235, 269, int(max_peso)
                )
                
                if not df_cargado.empty:
                    # Usar columna 'Volumen' o calcular matemáticamente
                    if 'Volumen' in df_cargado.columns:
                        df_cargado['m3'] = df_cargado['Volumen']
                    else:
                        df_cargado['m3'] = (df_cargado['Largo']*df_cargado['Ancho']*df_cargado['Alto'])/1e6
                        
                    vol_total = df_cargado['m3'].sum()
                    
                    # RESTRICCIÓN DE VOLUMEN MÍNIMO
                    if vol_total < min_vol:
                        break 
                    
                    contenedores_res[f"Contenedor {cont_global_idx}"] = {
                        "items": df_cargado,
                        "peso_total": peso,
                        "m3_total": vol_total,
                        "cg_x": coords_cg[0],
                        "cg_y": coords_cg[1],
                        "pedidos": [pedido_key]
                    }
                    
                    ids_cargados_total.update(ids)
                    items_pendientes_pedido = items_pendientes_pedido[~items_pendientes_pedido['ID'].isin(ids)]
                    cont_global_idx += 1
                else:
                    break
            
            progreso.progress(grupo_actual_idx / total_grupos)

    status.success(f"✅ ¡Planificación completada! Se generaron {len(contenedores_res)} contenedores.")
    progreso.empty()
    
    # Calculamos sobrantes
    items_sobrantes = df_total[~df_total['ID'].isin(ids_cargados_total)]
    
    return contenedores_res, items_sobrantes
import io

# ==============================================================================
# 6. FUNCIÓN DE EXPORTACIÓN A EXCEL (MULTIPLE HOJA)
# ==============================================================================
def generar_excel_descarga(resultados_dict, df_sobrante):
    output = io.BytesIO()
    # Usamos XlsxWriter como motor para soportar múltiples hojas
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        
        # 1. Hoja de Resumen de Contenedores
        resumen_data = []
        for nombre, data in resultados_dict.items():
            resumen_data.append({
                "Contenedor": nombre,
                "Peso Total (kg)": data['peso_total'],
                "Volumen (m3)": data['m3_total'],
                "CoG X": data['cg_x'],
                "CoG Y": data['cg_y'],
                "Cant. Paquetes": len(data['items'])
            })
        
        df_resumen = pd.DataFrame(resumen_data)
        df_resumen.to_excel(writer, sheet_name='Resumen_Flota', index=False)
        
        # 2. Hoja por cada Contenedor con detalle completo
        for nombre, data in resultados_dict.items():
            # Limpiamos el nombre para que sea válido como pestaña de Excel
            sheet_name = nombre.replace(" ", "_")[:31]
            data['items'].to_excel(writer, sheet_name=sheet_name, index=False)
            
        # 3. Hoja de Sobrantes
        if not df_sobrante.empty:
            df_sobrante.to_excel(writer, sheet_name='Sobrantes', index=False)
            
    return output.getvalue()

# ==============================================================================
# INTEGRACIÓN EN LA INTERFAZ
# ==============================================================================
if 'res' in st.session_state:
    resultados, sobrante = st.session_state['res']
    
    if resultados:
        st.divider()
        st.subheader("📥 Exportar Planificación")
        
        # Generar el archivo en memoria
        excel_data = generar_excel_descarga(resultados, sobrante)
        
        st.download_button(
            label="Descargar Plan de Estiba (Excel)",
            data=excel_data,
            file_name=f"Plan_Estiba_SVTI_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            icon="📊"
        )
# ==============================================================================
# 4. INTERFAZ DE USUARIO (SIDEBAR + EJECUCIÓN)
# ==============================================================================
with st.sidebar:
    st.header("1. Carga de Datos")
    uploaded_file = st.file_uploader("Excel (Paquete, Largo, Pedido, Pos...)", type=["xlsx"])
    
    df_clean = pd.DataFrame()
    seleccion_usuario = []
    btn_calc = False
    
    # Procesamos el archivo INMEDIATAMENTE para obtener las opciones del selector
    if uploaded_file:
        try:
            df_raw = pd.read_excel(uploaded_file)
            # Usamos la función de limpieza que definimos antes
            df_clean = limpiar_y_preparar_datos(df_raw)
            
            st.header("2. Configuración de Carga")
            
            # Crear lista de opciones legible: "Pedido: 100 - Pos: 1"
            # Mapeamos etiqueta -> Pedido_Key para filtrar después
            opciones_unicas = df_clean[['Pedido', 'Pos Pedido', 'Pedido_Key']].drop_duplicates().sort_values(['Pedido', 'Pos Pedido'])
            opciones_unicas['Label'] = "Pedido: " + opciones_unicas['Pedido'].astype(str) + " | Pos: " + opciones_unicas['Pos Pedido'].astype(str)
            
            # SELECTOR MULTIPLE
            opciones_display = opciones_unicas['Label'].tolist()
            seleccion_labels = st.multiselect(
                "Seleccionar Pedidos a Consolidar:",
                options=opciones_display,
                default=opciones_display # Por defecto selecciona todo
            )
            
            # Recuperar las Keys de la selección
            seleccion_keys = opciones_unicas[opciones_unicas['Label'].isin(seleccion_labels)]['Pedido_Key'].tolist()
            
            st.divider()
            max_w = st.number_input("Peso Máx por Contenedor (kg)", value=26000)
            
            # --- NUEVOS INPUTS AÑADIDOS ---
            col1, col2 = st.columns(2)
            with col1:
                min_v = st.number_input("Volumen Mínimo (m³)", value=10.0, step=1.0)
            with col2:
                num_cont = st.number_input("Cant. Contenedores", min_value=0, value=0, step=1, help="0 = Automático")
            
            if num_cont == 0:
                st.info(f"Modo Automático: Se generarán los contenedores necesarios. Se descartarán aquellos con < {min_v} m³.")
            else:
                st.info(f"Modo Fijo: La carga se distribuirá equitativamente en {num_cont} contenedores.")
            
            btn_calc = st.button("🚀 Calcular Carga", type="primary", disabled=(len(seleccion_keys)==0))
            
        except Exception as e:
            st.error(f"Error leyendo archivo: {e}")

# LÓGICA DE EJECUCIÓN
if uploaded_file and btn_calc and not df_clean.empty:
    # 1. Filtramos el DataFrame con la selección del usuario
    df_procesar = df_clean[df_clean['Pedido_Key'].isin(seleccion_keys)]
    
    if df_procesar.empty:
        st.warning("No hay ítems en la selección realizada.")
    else:
        # 2. Ejecutamos la optimización pasando los nuevos parámetros
        st.session_state['res'] = ejecutar_optimizacion_flota(
            df_total=df_procesar, 
            max_peso=max_w, 
            min_vol=min_v, 
            num_contenedores_fijos=num_cont
        )

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

# ✅ NUEVA LÓGICA DINÁMICA (Va justo después de que termina la función)
            pisos_unicos = sorted(df_items['Piso'].unique())
            columnas_pisos = st.columns(len(pisos_unicos))
            
            for idx, nombre_piso in enumerate(pisos_unicos):
                df_piso_actual = df_items[df_items['Piso'] == nombre_piso]
                
                with columnas_pisos[idx]: 
                    st.plotly_chart(
                        plot_2d_interactivo(df_piso_actual, f"📦 {nombre_piso}"), 
                        use_container_width=True
                    )
                    
            with tabs[2]: st.dataframe(df_items)

    if not sobrante.empty:
        st.error(f"⚠️ Quedaron {len(sobrante)} bultos sin cargar.")

















