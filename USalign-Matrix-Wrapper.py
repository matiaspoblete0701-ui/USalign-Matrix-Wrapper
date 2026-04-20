import glob
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.cluster.hierarchy import linkage, dendrogram
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
    parser.add_argument("-o", "--output", type=str, help="Prefijo para los archivos")
    parser.add_argument("-d", "--outdir", type=str, default="resultados", help="Carpeta de salida")
    return parser.parse_args()

def guardar_matrices_csv(m_sim, m_dist, etiquetas, args):
    if args.output:
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
        
        if args.output: 
            nombre_final = os.path.join(args.outdir, f"{args.output}_{nombre_base}.png")
            plt.savefig(nombre_final, dpi=300)
            print(f"Heatmap guardado en: {nombre_final}")
        
        plt.show()

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
    
    if args.output:
        nombre_final = os.path.join(args.outdir, f"{args.output}_clustermap.png")
        g.savefig(nombre_final, dpi=300, bbox_inches='tight')
        print(f"Clustermap guardado en: {nombre_final}")
        
    plt.show()

def main():
    args = definir_argumentos() 
    if args.output:
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

    print(f"Procesando {n} estructuras...")
    m_sim = np.ones((n, n)) 
    tiempo_total = 0 
    
    for i in range(n):
        for j in range(i+1, n):
            s1, s2, tiempo = obtener_tm_score(protein_files[i], protein_files[j])
            m_sim[i][j], m_sim[j][i] = s1, s2
            tiempo_total += tiempo 

    print(f"El tiempo de ejecución fue de {tiempo_total:.2f} segundos -> {tiempo_total//60:.0f} minutos con {tiempo_total%60:.2f} segundos.")
    # Cálculo corregido para el número exacto de comparaciones (n*(n-1)/2)
    print(f"Con un promedio de {tiempo_total/((n*(n-1))/2):.2f} segundos por cálculo.")

    m_sim_s = (m_sim + m_sim.T) / 2 
    m_dist = 1 - m_sim_s
    np.fill_diagonal(m_dist, 0) 
    etiquetas = [os.path.basename(p).split('.')[0] for p in protein_files]

    # --- GUARDADO DE DATOS CRUDOS ---
    guardar_matrices_csv(m_sim_s, m_dist, etiquetas, args)

    # --- CÁLCULO DEL AGRUPAMIENTO ---
    cond_dist = scipy.spatial.distance.squareform(m_dist)
    agrup = linkage(cond_dist, method="average")

    # --- GENERACIÓN DE GRÁFICOS ---
    print("Generando Heatmaps...")
    generar_heat_maps(m_sim_s, m_dist, etiquetas, args)

    print("Generando Dendrograma...")
    dist_max = max(agrup[:, 2])
    umbral = dist_max * 0.70
    plt.figure(figsize=(12, max(10, n * 0.5)))
    dendrogram(agrup, labels=etiquetas, orientation='right', color_threshold=umbral)
    plt.axvline(x=umbral, color='r', linestyle='--', label=f'Umbral ({umbral:.3f})')
    plt.legend()
    
    if args.output:
        nombre_dendro = os.path.join(args.outdir, f"{args.output}_dendrograma.png")
        plt.savefig(nombre_dendro, dpi=300, bbox_inches='tight')
        print(f"Dendrograma guardado en: {nombre_dendro}")
        
    # Restaurado el plt.show()
    plt.show()

    print("Generando Clustermap...")
    generar_clustermap(m_sim_s, agrup, etiquetas, args)
    
    if args.output:
        print(f"\nLos resultados están en la carpeta: {args.outdir}")

if __name__ == "__main__":
    main()