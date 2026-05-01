import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Configuración visual para que las figuras esten tipo artículo científico
plt.style.use('seaborn-v0_8-whitegrid')
os.makedirs("figures", exist_ok=True)

SEEDS = [42, 101, 202, 303, 404]
SWEEP_SEEDS = [42, 101, 202] # Las semillas que usamos
WINDOW_SIZE = 10  

# Retorno máximo objetivo: 80.
TARGET_RETURN = 80.0 

def load_seed_data(prefix, seeds_to_use=SEEDS):
    """Carga y agrupa los datos de las semillas para un experimento dado."""
    dfs = []
    for seed in seeds_to_use:
        path = f"logs/{prefix}_seed_{seed}.csv"
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
    if not dfs: return None
    
    combined = pd.concat(dfs)
    mean_df = combined.groupby("iteration").mean()
    std_df = combined.groupby("iteration").std()
    
    # Extraemos el último 10% de iteraciones para la convergencia
    final_10_percent = max(1, int(len(dfs[0]) * 0.1))
    final_scores = [df["mean_population_return"].tail(final_10_percent).mean() for df in dfs]
    
    return mean_df, std_df, final_scores, dfs

def generar_metricas_y_curvas():
    print("\n" + "="*65)
    print("1. RENDIMIENTO, ABLACIÓN Y SIGNIFICACIÓN ESTADÍSTICA")
    print("="*65)
    
    hybrid_data = load_seed_data("ablation_a_hybrid")
    ppo_data = load_seed_data("ablation_b_ppo")
    pso_data = load_seed_data("ablation_c_pso") # Añadido para el boxplot
    
    if not hybrid_data or not ppo_data:
        print("Error: Faltan datos de ablación. Ejecuta main.py primero.")
        return

    h_mean, h_std, h_final_scores, h_dfs = hybrid_data
    p_mean, p_std, p_final_scores, p_dfs = ppo_data

    # PUNTUACIÓN DE CONVERGENCIA (10% final) 
    h_conv, h_conv_std = np.mean(h_final_scores), np.std(h_final_scores)
    p_conv, p_conv_std = np.mean(p_final_scores), np.std(p_final_scores)
    
    print(f"[Convergencia] Híbrido (PSO+PPO): {h_conv:.2f} ± {h_conv_std:.2f}")
    print(f"[Convergencia] Línea Base (PPO):   {p_conv:.2f} ± {p_conv_std:.2f}")
    
    # MEJORA RELATIVA 
    if p_conv != 0:
        mejora = (h_conv - p_conv) / abs(p_conv) * 100
        print(f"[Mejora Relativa]: {mejora:.2f}% (∆ = (Rhibrido - Rbase)/|Rbase|)")
    
    # SIGNIFICACIÓN ESTADÍSTICA (T de Welch) 
    t_stat, p_value = stats.ttest_ind(h_final_scores, p_final_scores, equal_var=False)
    sig_text = "Significativo (Rechazamos H0)" if p_value < 0.05 else "NO Significativo"
    print(f"[T de Welch]: p-value = {p_value:.4f} -> {sig_text}")

    # EFICIENCIA MUESTRAL 
    h_smooth = h_mean["mean_population_return"].rolling(WINDOW_SIZE, min_periods=1).mean()
    p_smooth = p_mean["mean_population_return"].rolling(WINDOW_SIZE, min_periods=1).mean()
    
    h_target_idx = h_smooth[h_smooth >= TARGET_RETURN].first_valid_index()
    p_target_idx = p_smooth[p_smooth >= TARGET_RETURN].first_valid_index()
    
    h_timesteps = h_mean.loc[h_target_idx, "timesteps"] if h_target_idx is not None else "No alcanzó"
    p_timesteps = p_mean.loc[p_target_idx, "timesteps"] if p_target_idx is not None else "No alcanzó"
    print(f"[Eficiencia Muestral] Interacciones para alcanzar {TARGET_RETURN}:")
    print(f"   -Híbrido:   {h_timesteps}")
    print(f"   -Línea Base: {p_timesteps}")
    
    # ESTABILIDAD (Desviación Estándar Móvil Final) 
    h_stab = h_mean["mean_population_return"].tail(WINDOW_SIZE).std()
    p_stab = p_mean["mean_population_return"].tail(WINDOW_SIZE).std()
    print(f"[Estabilidad] Desviación estándar de los últimos {WINDOW_SIZE} episodios:")
    print(f"   -Híbrido: {h_stab:.2f} (Valores bajos indican que no oscila/no hay deriva)")
    print(f"   -Línea Base: {p_stab:.2f}")

    # Gráfico 1
    plt.figure(figsize=(10, 6))
    x_h = h_mean["timesteps"]
    x_p = p_mean["timesteps"]
    
    # Híbrido
    plt.plot(x_h, h_smooth, label="Híbrido (PSO+PPO)", color="blue", linewidth=2)
    h_std_smooth = h_std["mean_population_return"].rolling(WINDOW_SIZE, min_periods=1).mean()
    plt.fill_between(x_h, h_smooth - h_std_smooth, h_smooth + h_std_smooth, color="blue", alpha=0.2)
    
    # Línea Base
    plt.plot(x_p, p_smooth, label="Línea Base (PPO simple)", color="red", linestyle="--", linewidth=2)
    p_std_smooth = p_std["mean_population_return"].rolling(WINDOW_SIZE, min_periods=1).mean()
    plt.fill_between(x_p, p_smooth - p_std_smooth, p_smooth + p_std_smooth, color="red", alpha=0.1)

    plt.axhline(TARGET_RETURN, color="green", linestyle=":", label=f"Retorno Objetivo ({TARGET_RETURN})")
    plt.title("Curva de Aprendizaje (Suavizada): Ablación del Componente Bioinspirado", fontsize=14)
    plt.xlabel("Interacciones con el entorno (Timesteps)", fontsize=12)
    plt.ylabel("Retorno Medio por Episodio", fontsize=12)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("figures/1_curva_aprendizaje.pdf") 
    plt.close()
    print("Gráfico Curva de aprendizaje guardado (1_curva_aprendizaje.pdf)")

    # Gráfico 4 (boxplot de ablación)
    boxplot_data = []
    if hybrid_data:
        for score in hybrid_data[2]: boxplot_data.append({"Modelo": "Híbrido (PSO+PPO)", "Retorno": score})
    if ppo_data:
        for score in ppo_data[2]: boxplot_data.append({"Modelo": "Solo PPO", "Retorno": score})
    if pso_data:
        for score in pso_data[2]: boxplot_data.append({"Modelo": "Solo PSO", "Retorno": score})

    if boxplot_data:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=pd.DataFrame(boxplot_data), x="Modelo", y="Retorno", hue="Modelo", palette="Set2", legend=False)
        plt.title("Distribución del Retorno Final por Configuración (5 semillas)", fontsize=14)
        plt.ylabel("Retorno Medio Final (Último 10%)", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("figures/4_boxplot_ablacion.pdf")
        plt.close()
        print("Gráfico Boxplot guardado (4_boxplot_ablacion.pdf)")

def diagnosticos_y_costes():
    print("\n" + "="*65)
    print("2. DIAGNÓSTICOS BIOINSPIRADOS Y COSTE COMPUTACIONAL")
    print("="*65)
    
    data = load_seed_data("ablation_a_hybrid")
    if not data: return
    h_mean, h_std, _, _ = data
    
    # Gráfica 2: Dinámica Interna Velocidades
    plt.figure(figsize=(10, 4))
    plt.plot(h_mean.index, h_mean["mean_velocity"], color="purple", linewidth=2)
    plt.fill_between(h_mean.index, 
                     h_mean["mean_velocity"] - h_std["mean_velocity"], 
                     h_mean["mean_velocity"] + h_std["mean_velocity"], 
                     color="purple", alpha=0.2)
    plt.title("Dinámica Interna: Evolución de la Velocidad de Partículas (PSO)", fontsize=14)
    plt.xlabel("Episodio de Entrenamiento", fontsize=12)
    plt.ylabel("Magnitud de Velocidad (Pesos)", fontsize=12)
    plt.tight_layout()
    plt.savefig("figures/2_dinamica_interna.pdf")
    plt.close()
    
    # Coste Computacional 
    total_time = h_mean["time_total_cumulative"].iloc[-1]
    pso_time = h_mean["time_pso_cumulative"].iloc[-1]
    fraction = (pso_time / total_time) * 100 if total_time > 0 else 0
    
    print(f"[Coste] Tiempo total de entrenamiento: {total_time:.2f} s")
    print(f"[Coste] Tiempo dedicado a PSO: {pso_time:.2f} s")
    print(f"[Coste] Fracción del tiempo del componente híbrido: {fraction:.2f}%")
    print("\n[Acoplamiento] Frecuencia: El componente interactúa en cada actualización de política (100%).")

    # Gráfico 5 (coste temporal)
    plt.figure(figsize=(10, 6))
    plt.plot(h_mean.index, h_mean['time_total_cumulative'], label="Tiempo Total (PPO + PSO + Entorno)", color='black', lw=2)
    plt.plot(h_mean.index, h_mean['time_pso_cumulative'], label="Sobrecarga PSO (Álgebra matricial)", color='red', lw=2)
    plt.fill_between(h_mean.index, h_mean['time_pso_cumulative'], color='red', alpha=0.3)
    plt.title("Análisis de Coste Computacional: Tiempo vs Iteraciones", fontsize=14)
    plt.xlabel("Iteraciones de Entrenamiento", fontsize=12)
    plt.ylabel("Tiempo Acumulado (segundos)", fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("figures/5_coste_temporal.pdf")
    plt.close()
    print("Gráfico Coste Temporal guardado (5_coste_temporal.pdf)")

def graficas_barridos_y_sensibilidad():
    print("\n" + "="*65)
    print("3. BARRIDOS DE HIPERPARÁMETROS Y TABLA DE SENSIBILIDAD")
    print("="*65)
    
    # Gráfico 3: Gráficas de Barrido en Línea 
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Barrido N (Estructural)
    n_vals = [4, 8, 16, 32]
    scores_n = []
    for n in n_vals:
        data = load_seed_data(f"sweep_n_{n}", SWEEP_SEEDS)
        if data:
            scores_n.append(np.mean(data[2]))
        else:
            scores_n.append(np.nan)
    axes[0].plot(n_vals, scores_n, marker='o', linestyle='-', color='teal')
    axes[0].set_title("Sensibilidad: Tamaño Población (n)")
    axes[0].set_xlabel("Número de partículas")
    axes[0].set_ylabel("Puntuación Convergencia")
    
    # Barrido W (Bioinspirado)
    w_vals = [0.4, 0.7, 0.9]
    scores_w = []
    for w in w_vals:
        data = load_seed_data(f"sweep_w_{w}", SWEEP_SEEDS)
        if data: scores_w.append(np.mean(data[2]))
        else: scores_w.append(np.nan)
    axes[1].plot(w_vals, scores_w, marker='s', linestyle='-', color='darkorange')
    axes[1].set_title("Sensibilidad: Peso Inercia (w)")
    axes[1].set_xlabel("Valor de Inercia")
    
    # Barrido C (Bioinspirado)
    c_vals = [0.5, 1.0, 2.0]
    scores_c = []
    for c in c_vals:
        data = load_seed_data(f"sweep_c_{c}", SWEEP_SEEDS)
        if data: scores_c.append(np.mean(data[2]))
        else: scores_c.append(np.nan)
    axes[2].plot(c_vals, scores_c, marker='^', linestyle='-', color='firebrick')
    axes[2].set_title("Sensibilidad: Coeficientes Cognitivo/Social (c)")
    axes[2].set_xlabel("Valor de c1=c2")

    plt.tight_layout()
    plt.savefig("figures/3A_graficas_barridos.pdf")
    plt.close()
    
    # Gráfico 3: Mapa de sensibilidad
    results = np.zeros((len(c_vals), len(w_vals)))
    
    for i, c in enumerate(c_vals):
        for j, w in enumerate(w_vals):
            # Usar los datos de las corridas existentes para formar la tabla (w=0.7 y c=1.0 eran base)
            path_w = f"logs/sweep_w_{w}_seed_{SWEEP_SEEDS[0]}.csv"
            path_c = f"logs/sweep_c_{c}_seed_{SWEEP_SEEDS[0]}.csv"
            
            if os.path.exists(path_w) and c == 1.0: 
                df = pd.read_csv(path_w)
                results[i, j] = df["mean_population_return"].tail(6).mean()
            elif os.path.exists(path_c) and w == 0.7: 
                df = pd.read_csv(path_c)
                results[i, j] = df["mean_population_return"].tail(6).mean()
            else:
                results[i, j] = np.nan 

    plt.figure(figsize=(7, 5))
    sns.heatmap(results, annot=True, fmt=".1f", cmap="magma", 
                xticklabels=w_vals, yticklabels=c_vals, 
                cbar_kws={'label': 'Puntuación Convergencia'})
    plt.title("Tabla de Sensibilidad Bioinspirada", fontsize=14)
    plt.xlabel("Peso de Inercia PSO (w)", fontsize=12)
    plt.ylabel("Coeficientes (c)", fontsize=12)
    plt.tight_layout()
    plt.savefig("figures/3B_mapa_sensibilidad.pdf")
    plt.close()
    
    print("Gráficos de barridos guardados (3A_graficas_barridos.pdf)")
    print("Mapa de calor guardado (3B_mapa_sensibilidad.pdf)")

if __name__ == "__main__":
    print("Leyendo archivos .csv y calculando estadísticas...\n")
    generar_metricas_y_curvas()
    diagnosticos_y_costes()
    graficas_barridos_y_sensibilidad()
    print("\n¡Análisis completo! Las figuras se han guardado en la carpeta /figures")