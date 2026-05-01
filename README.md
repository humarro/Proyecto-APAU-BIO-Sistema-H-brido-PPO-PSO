# Proyecto: Sistema Híbrido Bioinspirado (PSO + PPO)

**Asignatura:** Aprendizaje Automático Bioinspirado (APBIO)  
**Universidad:** Universidad de Vigo (Curso 2025-2026)

Este repositorio contiene la implementación y los resultados de un sistema híbrido de Aprendizaje por Refuerzo que acopla Optimización por Enjambre de Partículas (PSO) con Proximal Policy Optimization (PPO), aplicado al entorno de control continuo `MountainCarContinuous-v0` de Gymnasium. El objetivo del componente bioinspirado (PSO) es inyectar diversidad y exploración global en el bucle de optimización de la política para evitar el colapso prematuro en un entorno caracterizado por recompensas dispersas.

---

## (a) Instrucciones de instalación

Para ejecutar este proyecto, se requiere **Python 3.9 o superior**. Se recomienda utilizar un entorno virtual.

1. Clonar el repositorio y acceder a la carpeta del proyecto:
    ```bash
    git clone <URL_DE_TU_REPOSITORIO_GITHUB>
    cd TRABAJO_APAU_BIO
    ```
2. Crear y activar un entorno virtual:
    ```bash
    python -m venv venv
    
    # En Windows:
    venv\Scripts\activate

    # En macOS/Linux:
    source venv/bin/activate
    ```
3. Instalar las dependencias requeridas:
    ```bash
    pip install -r requirements.txt
    ```
## (b) Comando para reproducir el experimento principal

El código es completamente autocontenido y el experimento se divide en dos fases: la generación de los datos (entrenamiento) y el análisis de resultados (evaluación y gráficas).

1. **Para ejecutar los experimentos** ejecuta el siguiente comando en tu terminal:
    ```bash
    python PSO_PPO_Hybrid.py
    ```
2. **Para generar las gráficas y las métricas estadísticas** (una vez haya finalizado el entrenamiento del paso anterior), ejecuta:
    ```bash
    python plot_results.py
    ```
## (c) Salida esperada y tiempo de ejecución aproximado

**Tiempo de ejecución aproximado:**  
El script de entrenamiento simulará múltiples poblaciones a lo largo de diversas semillas y barridos. El tiempo de ejecución estimado es de **1 hora y media a 2 horas en total**. El script que procesa los datos y genera las imagenes tarda apenas unos segundos.

**Salida esperada:**  

1. **Tras ejecutar `PSO_PPO_Hybrid.py`:**
   * Se creará un directorio `logs/` en la raíz del proyecto que contendrá archivos `.csv` con los registros en bruto para todos los experimentos de ablación y barridos.

   * En la consola se imprimirá el progreso iteración a iteración detallando la semilla actual.

2. **Tras ejecutar `plot_results.py`:**
   * En la consola se imprimirá el análisis estadístico completo.

   * Se creará un directorio `figures/` que contendrá las gráficas en formato PDF:

     * `1_curva_aprendizaje.pdf`: Comparativa de la eficiencia muestral y convergencia del híbrido frente a la línea base con bandas de varianza.
     * `2_dinamica_interna.pdf`: Evolución temporal de la velocidad de las partículas del PSO.
     * `3A_graficas_barridos.pdf`: Subgráficos mostrando la sensibilidad del rendimiento a los parámetros estructurales y bioinspirados.
     * `3B_mapa_sensibilidad.pdf`: Mapa de calor de la puntuación de convergencia.
     * `4_boxplot_ablacion.pdf`: Diagrama de cajas que evidencia la distribución y consistencia inter-semilla del rendimiento final.
     * `5_coste_temporal.pdf`: Análisis de coste computacional y sobrecarga de la heurística PSO respecto al tiempo total de ejecución.
