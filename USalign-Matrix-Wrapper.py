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
    parser = argparse.ArgumentParser(description="Análisis de Similitud Estructural")
    parser.add_argument("-r", "--ruta", nargs='+', required=True, help="Ruta(s) a carpetas con PDB/CIF")
    parser.add_argument("-o", "--output", type=str, required=True, help="Prefijo de salida")
    parser.add_argument("-d", "--outdir", type=str, required=True, help="Carpeta de salida")
    return parser.parse_args()

def optimizar_clustering(agrup, m_dist, n):
    alturas = agrup[:, 2]
    dist_max = alturas.max() if len(alturas) > 0 else 1.0
    umbral_minimo = dist_max * 0.25 

    def buscar_mejor_k(matriz_distancia, linkage_matrix, rango, u_min):
        mejor_s = -1
        mejor_k_local = 2
        n_local = len(matriz_distancia)
        
        for k in rango:
            if k >= n_local: break
            
            if k > 1:
                altura_corte = linkage_matrix[-k + 1, 2]
                if altura_corte < u_min and k > 3:
                    continue 

            labels = fcluster(linkage_matrix, k, criterion='maxclust')
            if len(set(labels)) < 2: continue
            
            score_base = silhouette_score(matriz_distancia, labels, metric='precomputed')
            # Factor de penalización para evitar sobresegmentación
            penalizacion = 1.0 - (k / (n_local * 2))
            score_adj = score_base * penalizacion
            
            if score_adj > mejor_s:
                mejor_s = score_adj
                mejor_k_local = k
        return mejor_k_local, mejor_s

    k_global, s_global = buscar_mejor_k(m_dist, agrup, range(2, min(n, 11)), umbral_minimo)
    
    if s_global < 0.45 and n > 15:
        labels_global = fcluster(agrup, k_global, criterion='maxclust')
        cluster_mayoritario = np.argmax(np.bincount(labels_global))
        indices_sub = np.where(labels_global == cluster_mayoritario)[0]

        if len(indices_sub) > 10:
            m_dist_sub = m_dist[np.ix_(indices_sub, indices_sub)]
            agrup_sub = linkage(scipy.spatial.distance.squareform(m_dist_sub), method="average")
            u_min_sub = agrup_sub[:, 2].max() * 0.2
            
            k_sub, s_sub = buscar_mejor_k(m_dist_sub, agrup_sub, range(2, min(len(indices_sub), 11)), u_min_sub)
            
            if s_sub > (s_global * 1.25):
                return agrup_sub[-k_sub + 1, 2], f"Nested_{k_sub}", s_sub

    umbral_final = alturas[-k_global + 1] if k_global > 1 else dist_max
    return umbral_final, k_global, s_global

def identificar_medoides(m_dist, etiquetas, labels):
    medoides = {}
    for cluster in np.unique(labels):
        indices = np.where(labels == cluster)[0]
        if len(indices) == 1:
            medoides[cluster] = indices[0]
        else:
            sub_dist = m_dist[np.ix_(indices, indices)]
            medoides[cluster] = indices[np.argmin(sub_dist.sum(axis=1))]
    return medoides

def exportar_visualizacion_pymol(medoides, protein_files, etiquetas, args):
    ruta_pml = os.path.join(args.outdir, f"{args.output}_pymol.pml")
    with open(ruta_pml, "w") as f:
        f.write("reinitialize\n")
        nombres = []
        for cluster, idx in medoides.items():
            path = os.path.abspath(protein_files[idx])
            name = f"Cluster_{cluster}_{etiquetas[idx]}"
            f.write(f"load {path}, {name}\n")
            nombres.append(name)
        if len(nombres) > 1:
            target = nombres[0]
            for i in range(1, len(nombres)):
                f.write(f"align {nombres[i]}, {target}\n")
        f.write("zoom\n")
    print(f"[*] Script de PyMOL guardado: {ruta_pml}")

def construir_newick(nodo, newick, parentdist, nombres_hojas):
    if nodo.is_leaf():
        return f"{nombres_hojas[nodo.id]}:{(parentdist - nodo.dist):.6f}{newick}"
    else:
        if len(newick) > 0:
            newick = f":{(parentdist - nodo.dist):.6f}{newick}"
        newick = f"({construir_newick(nodo.left, '', nodo.dist, nombres_hojas)},{construir_newick(nodo.right, '', nodo.dist, nombres_hojas)}){newick}"
        return newick

def main():
    args = definir_argumentos() 
    os.makedirs(args.outdir, exist_ok=True)
    
    protein_files = []
    for carpeta in args.ruta:
        if os.path.isdir(carpeta):
            for ext in ["*.pdb", "*.cif", "*.cif.gz", "*.pdb.gz"]:
                protein_files.extend(glob.glob(os.path.join(carpeta, ext)))
                
    protein_files = sorted(list(set(protein_files)))
    n = len(protein_files)
    if n < 2: return 
    
    m_sim = np.ones((n, n)) 
    total_comparaciones = (n * (n - 1)) // 2
    with tqdm(total=total_comparaciones, desc="TM-scores", unit="calc", colour="green") as pbar:
        for i in range(n):
           for j in range(i+1, n):
                s1, s2, _ = obtener_tm_score(protein_files[i], protein_files[j])
                m_sim[i][j], m_sim[j][i] = s1, s2
                pbar.update(1)
    
    m_sim_s = (m_sim + m_sim.T) / 2 
    m_dist = 1 - m_sim_s
    np.fill_diagonal(m_dist, 0) 
    etiquetas = [os.path.basename(p).split('.')[0] for p in protein_files]

    # Guardado de matrices
    ruta_sim = os.path.join(args.outdir, f"{args.output}_similitud.csv")
    pd.DataFrame(m_sim_s, index=etiquetas, columns=etiquetas).to_csv(ruta_sim)
    ruta_dist = os.path.join(args.outdir, f"{args.output}_distancia.csv")
    pd.DataFrame(m_dist, index=etiquetas, columns=etiquetas).to_csv(ruta_dist)
    print(f"[*] Matrices guardadas en: {args.outdir}")

    # Clustering
    agrup = linkage(scipy.spatial.distance.squareform(m_dist), method="average")
    umbral, k_final, score_final = optimizar_clustering(agrup, m_dist, n)
    print(f"\n[+] Resultado: {k_final} clusters encontrados (Silueta Ajustada: {score_final:.3f})")
    
    labels = fcluster(agrup, umbral, criterion='distance')
    medoides_dict = identificar_medoides(m_dist, etiquetas, labels)
    
    # Archivos adicionales
    ruta_res = os.path.join(args.outdir, f"{args.output}_clusters.csv")
    pd.DataFrame({
        "Proteina": etiquetas,
        "Cluster": labels,
        "Es_Medoide": [1 if i in medoides_dict.values() else 0 for i in range(n)]
    }).to_csv(ruta_res, index=False)
    print(f"[*] Clústeres y medoides guardados: {ruta_res}")

    exportar_visualizacion_pymol(medoides_dict, protein_files, etiquetas, args)

    arbol = to_tree(agrup, rd=False)
    with open(os.path.join(args.outdir, f"{args.output}_arbol.nwk"), "w") as f:
        f.write(construir_newick(arbol, "", arbol.dist, etiquetas) + ";")
    
    # Gráficos
    plt.figure(figsize=(12, max(8, n * 0.4)))
    dendrogram(agrup, labels=etiquetas, orientation='right', color_threshold=umbral)
    plt.axvline(x=umbral, color='r', linestyle='--', label=f'Umbral ({umbral:.3f})')
    plt.title(f"Dendrograma Estructural (K={k_final})")
    plt.savefig(os.path.join(args.outdir, f"{args.output}_dendrograma.pdf"), bbox_inches='tight')
    plt.close()

    sns.clustermap(m_sim_s, row_linkage=agrup, col_linkage=agrup, cmap="coolwarm", xticklabels=etiquetas, yticklabels=etiquetas)
    plt.savefig(os.path.join(args.outdir, f"{args.output}_clustermap.pdf"), bbox_inches='tight')
    plt.close()
    print(f"[!] Proceso finalizado exitosamente.")

if __name__ == "__main__":
    main()