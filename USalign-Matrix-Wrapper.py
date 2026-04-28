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
    proceso = subprocess.run(["./USalign", pdb1, pdb2, "-mm", "1", "-ter", "0"], capture_output=True, text=True)
    score1, score2 = 0.0, 0.0
    for linea in proceso.stdout.split('\n'):
        if linea.startswith("TM-score=") and "Structure_1" in linea:
            score1 = float(linea.split()[1])
        if linea.startswith("TM-score=") and "Structure_2" in linea:
            score2 = float(linea.split()[1])
            return score1, score2
    return score1, score2

def definir_argumentos():
    parser = argparse.ArgumentParser(description="Análisis Estructural Biológicamente Sensible")
    parser.add_argument("-r", "--ruta", nargs='+', required=True, help="Ruta(s) a carpetas con PDB/CIF")
    parser.add_argument("-o", "--output", type=str, required=True, help="Prefijo de salida")
    parser.add_argument("-d", "--outdir", type=str, required=True, help="Carpeta de salida")
    return parser.parse_args()

def optimizar_clustering(agrup, m_dist, n):
    alturas = agrup[:, 2]
    dist_max = alturas.max() if len(alturas) > 0 else 1.0
    umbral_minimo = 0.25 

    def buscar_mejor_k(matriz_distancia, linkage_matrix, rango):
        mejor_s = -1
        mejor_k_local = 2
        n_local = len(matriz_distancia)
        for k in rango:
            if k >= n_local: break
            labels = fcluster(linkage_matrix, k, criterion='maxclust')
            if len(set(labels)) < 2: continue
            score_base = silhouette_score(matriz_distancia, labels, metric='precomputed')
            penalizacion = 1.0 - (k / (n_local * 2)) 
            score_adj = score_base * penalizacion
            if score_adj > mejor_s:
                mejor_s = score_adj
                mejor_k_local = k
        return mejor_k_local, mejor_s

    k_global, s_global = buscar_mejor_k(m_dist, agrup, range(2, min(n, 12)))
    umbral_final = alturas[-k_global + 1] if k_global > 1 else dist_max
    
    if umbral_final < umbral_minimo:
        umbral_final = umbral_minimo
        labels_temp = fcluster(agrup, umbral_final, criterion='distance')
        k_global = len(np.unique(labels_temp))

    return umbral_final, k_global, s_global

def identificar_medoides(m_dist, labels):
    medoides = {}
    for cluster in np.unique(labels):
        indices = np.where(labels == cluster)[0]
        if len(indices) == 1:
            medoides[cluster] = indices[0]
        else:
            sub_dist = m_dist[np.ix_(indices, indices)]
            medoides[cluster] = indices[np.argmin(sub_dist.sum(axis=1))]
    return medoides

def exportar_visualizacion_pymol_por_cluster(medoides, labels, protein_files, etiquetas, args):
    for cluster, medoide_idx in medoides.items():
        indices_cluster = np.where(labels == cluster)[0]
        nombre_medoide = etiquetas[medoide_idx]
        ruta_pml = os.path.join(args.outdir, f"{args.output}_cluster_{cluster}_vs_{nombre_medoide}.pml")
        with open(ruta_pml, "w") as f:
            f.write("reinitialize\nset bg_rgb, [1, 1, 1]\n")
            path_med = os.path.abspath(protein_files[medoide_idx])
            f.write(f"load {path_med}, medoide_C{cluster}_{nombre_medoide}\n")
            f.write(f"color marine, medoide_C{cluster}_{nombre_medoide}\n")
            for idx in indices_cluster:
                if idx == medoide_idx: continue
                path_obj = os.path.abspath(protein_files[idx])
                obj_name = f"C{cluster}_{etiquetas[idx]}"
                f.write(f"load {path_obj}, {obj_name}\n")
                f.write(f"align {obj_name}, medoide_C{cluster}_{nombre_medoide}\n")
            f.write("zoom\nset ribbon_sampling, 1\nshow cartoon\n")
        print(f"[*] Script PyMOL guardado: {ruta_pml}")

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
                s1, s2 = obtener_tm_score(protein_files[i], protein_files[j])
                m_sim[i][j], m_sim[j][i] = s1, s2
                pbar.update(1)
    
    m_sim_s = (m_sim + m_sim.T) / 2 
    m_dist = 1 - m_sim_s
    np.fill_diagonal(m_dist, 0) 
    etiquetas = [os.path.basename(p).split('.')[0] for p in protein_files]

    # Guardado de matrices CSV
    ruta_sim = os.path.join(args.outdir, f"{args.output}_similitud.csv")
    ruta_dist = os.path.join(args.outdir, f"{args.output}_distancia.csv")
    pd.DataFrame(m_sim_s, index=etiquetas, columns=etiquetas).to_csv(ruta_sim)
    pd.DataFrame(m_dist, index=etiquetas, columns=etiquetas).to_csv(ruta_dist)
    print(f"[*] Matriz de similitud guardada en: {ruta_sim}")
    print(f"[*] Matriz de distancia guardada en: {ruta_dist}")

    agrup = linkage(scipy.spatial.distance.squareform(m_dist), method="average")
    umbral, k_final, score_final = optimizar_clustering(agrup, m_dist, n)
    labels = fcluster(agrup, umbral, criterion='distance')
    
    # Guardado de clústeres CSV
    ruta_cl = os.path.join(args.outdir, f"{args.output}_clusters.csv")
    pd.DataFrame({"Proteina": etiquetas, "Cluster": labels}).to_csv(ruta_cl, index=False)
    print(f"[*] Asignación de clústeres guardada en: {ruta_cl}")

    medoides_dict = identificar_medoides(m_dist, labels)
    exportar_visualizacion_pymol_por_cluster(medoides_dict, labels, protein_files, etiquetas, args)

    # Arbol Newick
    arbol = to_tree(agrup, rd=False)
    ruta_nwk = os.path.join(args.outdir, f"{args.output}_arbol.nwk")
    with open(ruta_nwk, "w") as f:
        f.write(construir_newick(arbol, "", arbol.dist, etiquetas) + ";")
    print(f"[*] Árbol Newick guardado en: {ruta_nwk}")

    # Dendrograma
    ruta_den = os.path.join(args.outdir, f"{args.output}_dendrograma.pdf")
    plt.figure(figsize=(12, max(8, n * 0.4)))
    dendrogram(agrup, labels=etiquetas, orientation='right', color_threshold=umbral)
    plt.axvline(x=umbral, color='r', linestyle='--', label=f'Umbral ({umbral:.3f})')
    plt.title(f"Dendrograma Estructural (K={len(np.unique(labels))})")
    plt.savefig(ruta_den, bbox_inches='tight')
    plt.close()
    print(f"[*] Dendrograma guardado en: {ruta_den}")

    # Heatmap
    ruta_hm = os.path.join(args.outdir, f"{args.output}_heatmap.pdf")
    plt.figure(figsize=(10, 10))
    sns.heatmap(m_sim_s, xticklabels=etiquetas, yticklabels=etiquetas, cmap="coolwarm", vmin=0, vmax=1)
    plt.savefig(ruta_hm, bbox_inches='tight')
    plt.close()
    print(f"[*] Heatmap de similitud guardado en: {ruta_hm}")

    # Clustermap
    ruta_cm = os.path.join(args.outdir, f"{args.output}_clustermap.pdf")
    g = sns.clustermap(m_sim_s, row_linkage=agrup, col_linkage=agrup, cmap="coolwarm", xticklabels=etiquetas, yticklabels=etiquetas)
    g.savefig(ruta_cm, bbox_inches='tight')
    plt.close()
    print(f"[*] Clustermap guardado en: {ruta_cm}")

    print(f"\n[!] Proceso finalizado. Se generaron {len(np.unique(labels))} clústeres.")

if __name__ == "__main__":
    main()