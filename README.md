# 🔭 PROJECT: LENS SENTINEL AI
### Agente Inteligente de Triaje para el Telescopio Espacial Nancy Grace Roman

Este repositorio contiene la implementación del software de escritorio **Lens Sentinel AI**, diseñado para optimizar la detección y validación de lentes gravitacionales (anillos de Einstein) en grandes volúmenes de datos astronómicos. 

---

## 1. Problemática

El avance de la astronomía observacional contemporánea, impulsado por instrumentos de última generación como el **Telescopio Espacial Nancy Grace Roman**, ha generado un cambio de paradigma en la adquisición de datos, transitando hacia la era del *Big Data* masivo.

*   **Saturación del Pipeline de Ingesta:** El volumen y la velocidad de las transmisiones de imágenes diarias (escala de Terabytes) sobresaturan las capacidades de almacenamiento y procesamiento de los centros de datos terrestres.
*   **Asimetría y Rareza Estructural:** Los fenómenos de alto valor científico, específicamente las **lentes gravitacionales** (Anillos de Einstein), presentan una tasa de incidencia extremadamente baja en el universo observable. Buscar estos patrones geométricos de forma manual equivale a resolver un problema de optimización con una densidad de recompensa casi nula.
*   **Fatiga Cognitiva y Costo Operacional:** El modelo tradicional *Human-in-the-Loop* (donde un astrónomo examina visualmente cada dataset) introduce un cuello de botella crítico. La revisión de millones de galaxias elípticas normales genera fatiga cognitiva, elevando la tasa de falsos negativos humanos y desperdiciando horas de alta especialización científica en tareas rutinarias de descarte.

---

## 2. Fundamentación

El diseño de este sistema inteligente no busca reemplazar el criterio del especialista, sino actuar como un **escudo computacional autónomo** fundamentado en los siguientes pilares de la ingeniería de software y el aprendizaje profundo:

*   **Desacoplamiento Estructural (Bajo Acoplamiento y Alta Cohesión):** La arquitectura separa estrictamente el *Módulo de Percepción* (red neuronal convolucional) del *Módulo de Razonamiento* (Agente de Triaje). Esto garantiza que la lógica de negocio (umbrales de decisión, políticas de archivado y gestión de colas) pueda modificarse o calibrarse de forma independiente sin necesidad de alterar, recompilar o reentrenar los pesos matemáticos de la red.
*   **Mitigación Científica del Desplazamiento de Dominio (*Domain Shift*):** Al entrenar inicialmente con datos mixtos (terrestres vs. espaciales), se descubrió que las redes profundas explotan atajos estadísticos en el ruido de fondo de las imágenes. La unificación del dataset bajo un único instrumento (*Pure Roman Dataset*) obliga a la arquitectura a aprender morfología física real (curvatura de arcos luminosos) en lugar de firmas de ruido locales.
*   **Filosofía de Programación Defensiva:** En sistemas críticos de clasificación, forzar una salida binaria $(0, 1)$ ante alta incertidumbre estadística introduce un riesgo inaceptable. El sistema implementa una **lógica ternaria** mediante un búfer de ambigüedad (*Revisión Secundaria*). Si los datos sufren degradación, el agente autogestiona el riesgo aislando la muestra en lugar de emitir un fallo catastrófico.

---

## 3. Metodología

El pipeline de ejecución del software se estructura de manera secuencial a través de componentes especializados y acoplados por interfaces de datos explícitas:

```
[ Ingesta HDF5 ] ➔ [ Normalización lineal ] ➔ [ CNN Classifier ] ➔ [ Decision Triage Agent ] ➔ [ Interfaz CustomTkinter ]
```

1.  **Fase 1: Ingesta y Extracción (`H5_Data_Extractor`):** El software realiza una lectura indexada y perezosa (*lazy loading*) del archivo de estructuras jerárquicas HDF5 (`.h5`), aislando los objetos de simulación mediante llaves únicas. Para mitigar sesgos de memoria, se aplicó una separación estricta de conjuntos, reservando un pozo ciego de datos de prueba independientes mediante una semilla aleatoria fija ($Seed = 777$).
2.  **Fase 2: Normalización y Mutación Morfológica (`Data_Standardizer`):** Cada matriz de flujo de fotones en punto flotante se somete a una normalización lineal de contraste en el rango clásico de píxeles:
    
    $$I_{norm} = \frac{I - I_{min}}{I_{max} - I_{min}} \times 255$$
    
    Para el modelado sintético de la clase negativa (Galaxias Normales), se aplicó un algoritmo de aislamiento central, recortando el **45% del centro de la imagen** para eliminar los arcos gravitacionales periféricos, reescalando el núcleo mediante remuestreo Lanczos a dimensiones de $128 \times 128 \times 1$ e inyectando ruido gaussiano cruzado controlado.
3.  **Fase 3: Módulo de Percepción (`CNN_Classifier`):** Se implementó una red neuronal convolucional secuencial compuesta por tres bloques de extracción de características. Cada bloque integra una capa `Conv2D` (filtros progresivos de 32, 64 y 128 con *kernel* de $3 \times 3$ y activación `ReLU`) y una capa `MaxPooling2D` para reducir dimensionalidad espacial. La clasificación final se procesa mediante una capa densa acoplada a un `Dropout` de seguridad de 0.5 y una función de activación de salida sigmoide:
    
    $$S(x) = \frac{1}{1 + e^{-x}}$$

4.  **Fase 4: Capa de Toma de Decisiones (`Intelligent_Triage_Agent`):** Componente de software encargado de evaluar el escalar continuo de probabilidad $P$ emitido por la CNN. Aplica una función por tramos condicional para determinar la acción operacional del sistema:
    *   $P \le 0.35$: Acción = `ARCHIVADO_AUTÓNOMO` (Descarte inmediato).
    *   $0.35 < P < 0.85$: Acción = `REVISIÓN_SECUNDARIA` (Aislamiento en Búfer).
    *   $P \ge 0.85$: Acción = `DESPACHO_URGENTE` (Alerta al especialista).
5.  **Fase 5: Despliegue e Interacción Gráfica (`CustomTkinter_Desktop_App`):** Construcción de un entorno local de escritorio conducido por eventos. Utiliza hilos de simulación asíncronos y no bloqueantes a través del método `.after()`, permitiendo visualizar la clasificación masiva en tiempo real sin congelar la tasa de refresco de la interfaz de usuario.

---

## 4. Resultados del Sistema

El sistema final fue sometido a una evaluación de calidad cuantitativa empleando un conjunto de prueba ciego de **1,500 muestras totalmente independientes** (750 galaxias normales sintéticas y 750 lentes gravitacionales reales).

### Matriz de Confusión (Módulo de Percepción - CNN)

| Real \ Pred | Pred Neg (Normal) | Pred Pos (Lente) |
| :--- | :---: | :---: |
| **Real Neg** | 750 (TN) | 0 (FP) |
| **Real Pos** | 0 (FN) | 750 (TP) |

### Decisiones Operacionales (Módulo de Razonamiento - Agente)

| Real \ Acción | Archivados (Archiv.) | En Búfer (Review) | Críticos (Urgent) |
| :--- | :---: | :---: | :---: |
| **Real Neg** | 749 | 1 | 0 |
| **Real Pos** | 0 | 0 | 750 |

### Análisis Métrico de Calidad de Software:
*   **Robustez del Módulo de Percepción:** Al operar en el mismo dominio óptico espacial de *Roman*, la CNN alcanzó un **Accuracy, Precision y Recall de 1.00 (100%)** sobre el set de prueba, demostrando una convergencia limpia y curvas de pérdida asintóticas estables.
*   **Garantía de Cero Pérdida Científica ($FNR = 0.00\%$):** La métrica de calidad de software más crítica del proyecto, la Tasa de Falsos Negativos (*False Negative Rate*), se ubicó estrictamente en **0.00%**. Ninguna lente real fue enviada a la columna de archivado por error, validando experimentalmente la seguridad del umbral inferior de $0.35$.
*   **Validación del Mecanismo de Programación Defensiva:** El agente demostró su utilidad operacional al aislar **exactamente 1 caso ambiguo** en la columna de *Revisión Secundaria*. El sistema identificó estadísticamente la incertidumbre de esa muestra específica y evitó una clasificación binaria forzada errónea.
*   **Optimización del Tiempo Experto:** El sistema automatizó por completo el descarte de **749 galaxias normales**, reduciendo la carga de revisión del astrónomo en aproximadamente un **50% del volumen total** del stream de datos.

---

## 5. Conclusiones

*   **Certeza Operacional:** La integración de redes convolucionales profundas como módulos de percepción acoplados a agentes de triaje basados en reglas demuestra ser una solución de software altamente viable, eficiente y segura para la gestión de macrodatos e imágenes astronómicas en tiempo real.
*   **Supremacía de la Ingeniería de Datos:** El rendimiento del sistema evidenció que la robustez matemática no depende exclusivamente de la complejidad o profundidad de la red neuronal, sino de la rigurosidad, homogeneidad y eliminación de sesgos de origen en el pipeline de preparación de los datasets.
*   **Simbiosis Arquitectónica Humano-IA:** El software consolida un modelo eficiente de asistencia científica. Al filtrar de forma autónoma el ruido de fondo y las galaxias irrelevantes del universo, el sistema actúa como un optimizador de recursos que permite a los especialistas enfocar su capacidad analítica exclusivamente en la confirmación de descubrimientos de alto impacto.

---

## 6. Requisitos e Instalación

Para ejecutar la aplicación localmente, asegúrate de contar con una versión de **Python compatible (entre 3.9 y 3.13)**. 

> [!IMPORTANT]
> **TensorFlow** no es compatible actualmente con **Python 3.14**. Si tu versión predeterminada del sistema es Python 3.14+, debes forzar la creación del entorno virtual con una versión compatible (por ejemplo, Python 3.13).

### 1. Crear el Entorno Virtual

Crea el entorno virtual especificando una versión compatible:

*   **Windows (usando el Python Launcher `py`):**
    ```powershell
    py -3.13 -m venv env
    ```
*   **macOS / Linux / Unix (o si tu comando `python` ya apunta a una versión compatible):**
    ```bash
    python3 -m venv env
    ```

> [!TIP]
> **¿Error `Unable to copy...` en Windows?**
> Si el comando falla indicando que no se pudo copiar `venvlauncher.exe` o `python.exe`, significa que un proceso (como tu editor de código, una terminal activa o el corrector de sintaxis/language server de Python) está utilizando el entorno virtual actual y tiene bloqueado el archivo. Cierra tu editor (como VS Code) y cualquier otra consola, y vuelve a intentar el comando.

### 2. Activar el Entorno Virtual

Activa el entorno según tu sistema y consola:

*   **Windows (PowerShell):**
    ```powershell
    .\env\Scripts\Activate.ps1
    ```
*   **Windows (Símbolo del sistema / CMD):**
    ```cmd
    env\Scripts\activate.bat
    ```
*   **macOS / Linux / Git Bash:**
    ```bash
    source env/bin/activate
    ```

### 3. Instalar Dependencias

Con el entorno virtual activo, instala las dependencias necesarias:

```bash
pip install numpy tensorflow customtkinter pillow
```

### Ejecución de la aplicación:
```bash
python app.py
```
