import glob
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import scipy
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, to_tree
from sklearn.metrics import silhouette_score
import os
import time
import argparse
import pandas as pd

def obtener_tm_score(pdb1, pdb2):
    inicio = time.perf_counter()
    proceso = subprocess.run(["./USalign", pdb1, pdb2, "-mm", "1", "-ter", "0"], capture_output=True, text=True)
    fin = time.perf_counter()
    tiempo = fin - inicio
    score1, score2 = 0.0, 0.0
    for linea in proceso.stdout.split('\n'):
        if linea.startswith("TM-score=") and "Structure_1" in linea:
            score1 = float(linea.split()[1])
        if linea.startswith("TM-score=") and "Structure_2" in linea:
            score2 = float(linea.split()[1])
            return score1, score2, tiempo
    return score1, score2, tiempo

def definir_argumentos():
    parser = argparse.ArgumentParser(description="Análisis de Similitud Estructural de Proteínas")
    parser.add_argument("-r", "--ruta", nargs='+', required=True,
                        help="Ruta(s) a las carpetas que contienen archivos .pdb o .cif")
    # Hacemos obligatorio el prefijo de salida
    parser.add_argument("-o", "--output", type=str, required=True, 
                        help="Prefijo para los archivos generados (Obligatorio)")
    # Hacemos obligatoria la carpeta de salida (se quita el default para forzar su uso)
    parser.add_argument("-d", "--outdir", type=str, required=True, 
                        help="Carpeta de salida (Obligatorio)")
    return parser.parse_args()

def optimizar_clustering(agrup, m_dist, n):
    """Prueba diferentes números de clusters y devuelve el mejor umbral penalizando solitarios."""
    mejor_score_ajustado = -1e9
    mejor_k = 2
    
    # Probamos un rango amplio (hasta 30 clusters)
    rango_k = range(2, min(n, 31)) 
    
    for k in rango_k:
        labels = fcluster(agrup, k, criterion='maxclust')
        base_score = silhouette_score(m_dist, labels, metric='precomputed')
        
        # --- LÓGICA DE PENALIZACIÓN ---
        # Contamos cuántas proteínas hay en cada cluster
        counts = np.bincount(labels)
        # Identificamos cuántos clusters tienen solo 1 proteína (outliers)
        num_solitarios = np.sum(counts == 1)
        
        # Penalizamos el score: a más clusters solitarios, menor es el puntaje.
        # Esto obliga al algoritmo a preferir un K donde las proteínas estén agrupadas.
        score_ajustado = base_score - (num_solitarios / k) * 0.3
        
        if score_ajustado > mejor_score_ajustado:
            mejor_score_ajustado = score_ajustado
            mejor_k = k
            
    # --- CÁLCULO DEL UMBRAL (Punto medio para asegurar colorización) ---
    alturas = agrup[:, 2]
    # Usamos el punto medio entre el salto del mejor_k y el anterior
    dist_k = alturas[-mejor_k + 1]
    dist_prev = alturas[-mejor_k] if mejor_k < n else dist_k
    umbral_dinamico = (dist_k + dist_prev) / 2
    
    # Calculamos el score real final para el reporte
    final_labels = fcluster(agrup, mejor_k, criterion='maxclust')
    score_final = silhouette_score(m_dist, final_labels, metric='precomputed')
    
    return umbral_dinamico, mejor_k, score_final

# --- NUEVAS FUNCIONES PARA NEWICK ---
def construir_newick(nodo, newick, parentdist, nombres_hojas):
    """Función recursiva para traducir nodos de SciPy a texto Newick."""
    if nodo.is_leaf():
        return f"{nombres_hojas[nodo.id]}:{(parentdist - nodo.dist):.6f}{newick}"
    else:
        if len(newick) > 0:
            newick = f":{(parentdist - nodo.dist):.6f}{newick}"
        newick = f"({construir_newick(nodo.left, '', nodo.dist, nombres_hojas)},{construir_newick(nodo.right, '', nodo.dist, nombres_hojas)}){newick}"
        return newick

def guardar_newick(agrup, etiquetas, args):
    """Genera y guarda el archivo .nwk"""
    arbol = to_tree(agrup, rd=False)
    cadena_newick = construir_newick(arbol, "", arbol.dist, etiquetas) + ";"
    ruta_newick = os.path.join(args.outdir, f"{args.output}_arbol.nwk")
    with open(ruta_newick, "w") as f:
        f.write(cadena_newick)
    print(f"Formato Newick guardado en: {ruta_newick}")
# ------------------------------------

def guardar_matrices_csv(m_sim, m_dist, etiquetas, args):
    ruta_sim = os.path.join(args.outdir, f"{args.output}_similitud.csv")
    ruta_dist = os.path.join(args.outdir, f"{args.output}_distancia.csv")
    
    df_sim = pd.DataFrame(m_sim, index=etiquetas, columns=etiquetas)
    df_dist = pd.DataFrame(m_dist, index=etiquetas, columns=etiquetas)
    
    df_sim.to_csv(ruta_sim)
    df_dist.to_csv(ruta_dist)
    print(f"Matrices CSV guardadas en: {args.outdir}")

def generar_heat_maps(m_sim, m_dist_sim, etiquetas, args):
    tareas = [
        (m_sim, "similitud", "coolwarm", 0, 1),
        (m_dist_sim, "distancia", "viridis", 0, 1)
    ]
    
    n_prot = len(etiquetas)
    lado_figura = max(10, n_prot * 0.5) 
    
    for matriz, nombre_base, mapa_color, v_min, v_max in tareas:
        plt.figure(figsize=(lado_figura, lado_figura)) 
        ax = sns.heatmap(
            matriz, 
            xticklabels=etiquetas, 
            yticklabels=etiquetas, 
            cmap=mapa_color, 
            vmin=v_min, 
            vmax=v_max,
            annot=False, 
            fmt=".3f"
        )    
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.tight_layout()
        
        # Guardado en PDF y cierre de figura
        nombre_final = os.path.join(args.outdir, f"{args.output}_{nombre_base}.pdf")
        plt.savefig(nombre_final, format='pdf', bbox_inches='tight')
        print(f"Heatmap guardado en: {nombre_final}")
        
        plt.close() # Reemplaza a plt.show()

def generar_clustermap(m_sim_s, agrup, etiquetas, args):
    n_prot = len(etiquetas)
    lado_figura = max(10, n_prot * 0.5) 
    
    g = sns.clustermap(
        m_sim_s,
        row_linkage=agrup,
        col_linkage=agrup,
        xticklabels=etiquetas,
        yticklabels=etiquetas,
        cmap="coolwarm", 
        vmin=0, 
        vmax=1,
        annot=False, 
        fmt=".3f",
        figsize=(lado_figura, lado_figura), 
        dendrogram_ratio=0.15,
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
        tree_kws={'colors': 'gray', 'linewidths': 1.5}
    )

    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Guardado en PDF y cierre de figura
    nombre_final = os.path.join(args.outdir, f"{args.output}_clustermap.pdf")
    g.savefig(nombre_final, format='pdf', bbox_inches='tight')
    print(f"Clustermap guardado en: {nombre_final}")
    
    plt.close() # Reemplaza a plt.show()
def encontrar_medoide(cluster_proteinas, m_dist, etiquetas):
    """Encuentra la proteína que es el centro (medoide) del cluster."""
    if len(cluster_proteinas) == 1:
        return cluster_proteinas[0]
    
    # Índices de las proteínas del cluster en la matriz original
    indices = [etiquetas.index(p) for p in cluster_proteinas]
    # Submatriz de distancias solo para este cluster
    submatriz = m_dist[np.ix_(indices, indices)]
    # La proteína con la menor suma de distancias a las demás es el medoide
    indice_medoide_local = np.argmin(submatriz.sum(axis=1))
    
    return cluster_proteinas[indice_medoide_local]

def generar_scripts_pymol(df_resultados, protein_files, m_dist, etiquetas, args):
    """Genera archivos .pml para visualizar cada cluster en PyMOL."""
    scripts_dir = os.path.join(args.outdir, "scripts_pymol")
    os.makedirs(scripts_dir, exist_ok=True)
    
    # Mapeo de nombre -> ruta absoluta para que PyMOL encuentre los archivos
    rutas_dict = {os.path.basename(f).split('.')[0]: os.path.abspath(f) for f in protein_files}

    for cluster_id in df_resultados['Cluster'].unique():
        prot_cluster = df_resultados[df_resultados['Cluster'] == cluster_id]['Proteina'].tolist()
        
        # Solo generamos script si el cluster tiene más de 1 proteína para alinear
        if len(prot_cluster) < 2: continue 
        
        medoide = encontrar_medoide(prot_cluster, m_dist, etiquetas)
        ruta_pml = os.path.join(scripts_dir, f"cluster_{cluster_id}.pml")
        
        with open(ruta_pml, "w") as f:
            f.write(f"# Script PyMOL - Cluster {cluster_id}\nreinitialize\n\n")
            # Cargar el medoide (estacionario)
            f.write(f"load {rutas_dict[medoide]}, {medoide}\n")
            f.write(f"color magenta, {medoide}\n")
            
            # Cargar y alinear el resto (móviles)
            for prot in prot_cluster:
                if prot == medoide: continue
                f.write(f"load {rutas_dict[prot]}, {prot}\n")
                f.write(f"align {prot}, {medoide}\n")
            
            f.write("\nshow cartoon\nutil.cbc\norient\n")

def main():
    args = definir_argumentos() 
    os.makedirs(args.outdir, exist_ok=True)

    protein_files = []
    for carpeta in args.ruta:
        if os.path.isdir(carpeta):
            protein_files.extend(glob.glob(os.path.join(carpeta, "*.pdb")) + 
                                glob.glob(os.path.join(carpeta, "*.cif")) +
                                glob.glob(os.path.join(carpeta, "*.cif.gz")) +
                                glob.glob(os.path.join(carpeta, "*.pdb.gz")))

    protein_files = sorted(list(set(protein_files)))
    n = len(protein_files)
    
    if n < 2:
        print("\n[!] ERROR: Se requieren al menos 2 archivos.")
        return 
    
    m_sim = np.ones((n, n)) 
    tiempo_total = 0 
    total_comparaciones = (n * (n - 1)) // 2
    print(f"Procesando {n} estructuras ({total_comparaciones} comparaciones totales)...")

    with tqdm(total=total_comparaciones, desc="Calculando TM-scores", unit="calc", colour="green") as pbar:
        for i in range(n):
           for j in range(i+1, n):
                s1, s2, tiempo = obtener_tm_score(protein_files[i], protein_files[j])
                m_sim[i][j], m_sim[j][i] = s1, s2
                tiempo_total += tiempo  
                pbar.update(1)
    
    m_sim_s = (m_sim + m_sim.T) / 2 
    m_dist = 1 - m_sim_s
    np.fill_diagonal(m_dist, 0) 
    etiquetas = [os.path.basename(p).split('.')[0] for p in protein_files]

    guardar_matrices_csv(m_sim_s, m_dist, etiquetas, args)

    cond_dist = scipy.spatial.distance.squareform(m_dist)
    agrup = linkage(cond_dist, method="average")

    print("\nOptimizando umbral con Silhouette Score...")
    umbral, k_optimo, score_optimo = optimizar_clustering(agrup, m_dist, n)
    
    # --- 1. OBTENER LABELS PARA PYMOL Y REPORTE ---
    labels_clusters = fcluster(agrup, k_optimo, criterion='maxclust')
    df_reporte = pd.DataFrame({"Proteina": etiquetas, "Cluster": labels_clusters})
    df_reporte.to_csv(os.path.join(args.outdir, f"{args.output}_resultados.csv"), index=False)

    # --- 2. GENERACIÓN DE GRÁFICOS ---
    generar_heat_maps(m_sim_s, m_dist, etiquetas, args)

    plt.figure(figsize=(12, max(10, n * 0.3)))
    dendrogram(agrup, labels=etiquetas, orientation='right', color_threshold=umbral, above_threshold_color='grey')
    plt.axvline(x=umbral, color='r', linestyle='--', label=f'Umbral Silhouette ({umbral:.3f})')
    plt.title(f"Dendrograma Estructural (K={k_optimo}, Silueta={score_optimo:.2f})")
    plt.legend()
    plt.savefig(os.path.join(args.outdir, f"{args.output}_dendrograma.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

    guardar_newick(agrup, etiquetas, args)
    generar_clustermap(m_sim_s, agrup, etiquetas, args)

    # --- 3. NUEVA SECCIÓN: GENERAR SCRIPTS DE PYMOL ---
    print("\n[+] Generando sesiones de PyMOL basadas en medoides...")
    generar_scripts_pymol(
        df_resultados=df_reporte, 
        protein_files=protein_files, 
        m_dist=m_dist, 
        etiquetas=etiquetas, 
        args=args
    )
    
    print(f"\n[!] Finalizado. Los resultados y scripts están en: {args.outdir}")


if __name__ == "__main__":
    main()