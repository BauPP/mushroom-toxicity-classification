# Fórmulas Matemáticas de los Modelos de Machine Learning
## Proyecto: Sistema Inteligente de Clasificación Micológica
**Dataset:** Secondary Mushroom Dataset (61,069 registros)  
**Variable Objetivo:** $Y \in \{0, 1\}$ (0 = Comestible / *Edible*, 1 = Venenoso / *Poisonous*)

---

## 1. Regresión Logística (`LogisticRegression`)

La Regresión Logística modela la probabilidad a posteriori de que un hongo sea venenoso ($Y=1$) condicionado a su vector de características preprocesadas $\mathbf{x} \in \mathbb{R}^d$.

### 1.1 Función de Hipótesis (Sigmoide / Logística)
$$P(Y=1 \mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

donde el predictor lineal $z$ se define como:
$$z = w_0 + \sum_{j=1}^d w_j x_j = \mathbf{w}^T \mathbf{x}$$

- $\mathbf{w} = [w_1, w_2, \dots, w_d]^T$ son los coeficientes o pesos del modelo.
- $w_0$ es el sesgo (*bias* o intercepto).

### 1.2 Función de Pérdida (Binary Cross-Entropy / Log-Loss con Regularización $L_2$)
Scikit-Learn optimiza la log-verosimilitud negativa con penalización Ridge ($L_2$):

$$J(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(\sigma(\mathbf{w}^T \mathbf{x}_i)) + (1 - y_i) \ln(1 - \sigma(\mathbf{w}^T \mathbf{x}_i)) \right] + \frac{1}{2C} \|\mathbf{w}\|_2^2$$

- $N$: Número de instancias en el conjunto de entrenamiento.
- $C$: Inverso de la fuerza de regularización ($C = \frac{1}{\lambda}$).

### 1.3 Regla de Decisión y Umbral Asimétrico ($\tau$)
$$\hat{y} = \begin{cases} 1 & \text{si } P(Y=1 \mid \mathbf{x}) \ge \tau \\ 0 & \text{si } P(Y=1 \mid \mathbf{x}) < \tau \end{cases}$$

Bajo el enfoque estándar $\tau = 0.50$. En nuestra **metodología de riesgo cero**, $\tau^*$ se calibra sobre el modelo que se despliega en producción: la Regresión Logística ilustra el efecto con $\tau^* < 0.50$; el modelo ganador (Random Forest) opera con $\tau^* = 0.52$, el centro de la franja sin errores de una validación cruzada 5-fold — ver §5.2.

![Sigmoide y Binary Cross-Entropy](figuras_modelo_matematico/esquema_sigmoide_bce.png)

---

## 2. Random Forest (`RandomForestClassifier`)

Random Forest es un ensamble de aprendizaje no paramétrico basado en **Bagging** (*Bootstrap Aggregating*) y subespacios aleatorios de características.

### 2.1 Criterio de Impureza de Gini (División de Nodos)
Para un nodo $t$ con proporción de clases $p_k(t) = \frac{N_{t, k}}{N_t}$ donde $k \in \{0, 1\}$:

$$I_G(t) = 1 - \sum_{k=0}^1 p_k(t)^2 = 1 - \left( p_0(t)^2 + p_1(t)^2 \right)$$

### 2.2 Ganancia de Información / Reducción de Impureza
Al dividir un nodo padre $t$ en un hijo izquierdo $t_L$ e hijo derecho $t_R$ mediante un corte de variable $\theta = (j, v)$:

$$\Delta I_G(t, \theta) = I_G(t) - \left( \frac{N_L}{N_t} I_G(t_L) + \frac{N_R}{N_t} I_G(t_R) \right)$$

El árbol selecciona el corte que maximiza $\Delta I_G(t, \theta)$ evaluando sobre un subconjunto aleatorio de $\sqrt{d}$ variables.

### 2.3 Agregación del Ensamble
Dado un bosque de $B$ árboles $\{T_1, T_2, \dots, T_B\}$:

- **Probabilidad agregada:**
  $$\hat{P}(Y=1 \mid \mathbf{x}) = \frac{1}{B} \sum_{b=1}^B P_b(Y=1 \mid \mathbf{x})$$

- **Predicción final por votación de mayoría:**
  $$\hat{y} = \text{moda} \left\{ T_1(\mathbf{x}), T_2(\mathbf{x}), \dots, T_B(\mathbf{x}) \right\}$$

![Árbol CART y agregación bagging de Random Forest](figuras_modelo_matematico/esquema_arbol_bosque.png)

---

## 3. XGBoost (`XGBClassifier`)

XGBoost (*Extreme Gradient Boosting*) es un algoritmo de ensamble secuencial que optimiza una función objetivo específica mediante árboles de regresión aditivos (CART).

### 3.1 Modelo Aditivo
La predicción acumulada en la iteración $t$ para una instancia $i$ es:
$$\hat{y}_i^{(t)} = \sum_{k=1}^t f_k(\mathbf{x}_i) = \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)$$
donde $f_t(\mathbf{x}) \in \mathcal{F}$ representa la estructura y pesos de las hojas del árbol en el paso $t$.

### 3.2 Función Objetivo Regularizada
$$\mathcal{L}^{(t)} = \sum_{i=1}^N l\left(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)\right) + \Omega(f_t)$$

El término de complejidad del árbol $\Omega(f_t)$ es:
$$\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$

- $T$: Número de hojas del árbol.
- $w_j$: Puntuación de peso (*leaf weight*) en la hoja $j$.
- $\gamma, \lambda$: Hiperparámetros de penalización de complejidad y regularización $L_2$.

### 3.3 Aproximación por Serie de Taylor (2° Orden)
$$\tilde{\mathcal{L}}^{(t)} \approx \sum_{i=1}^N \left[ l\left(y_i, \hat{y}_i^{(t-1)}\right) + g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \Omega(f_t)$$

Donde los gradientes de primer y segundo orden de la pérdida logística son:
$$g_i = \frac{\partial l(y_i, \hat{y}^{(t-1)})}{\partial \hat{y}^{(t-1)}} = \sigma(\hat{y}_i^{(t-1)}) - y_i$$
$$h_i = \frac{\partial^2 l(y_i, \hat{y}^{(t-1)})}{\partial (\hat{y}^{(t-1)})^2} = \sigma(\hat{y}_i^{(t-1)}) \left( 1 - \sigma(\hat{y}_i^{(t-1)}) \right)$$

### 3.4 Peso Óptimo de la Hoja y Ganancia de Corte (Split Gain)
- **Peso analítico óptimo en la hoja $j$:**
  $$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

- **Ganancia de división para evaluar cortes:**
  $$\text{Gain} = \frac{1}{2} \left[ \frac{\left( \sum_{i \in I_L} g_i \right)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{\left( \sum_{i \in I_R} g_i \right)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{\left( \sum_{i \in I} g_i \right)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma$$

![Modelo aditivo secuencial de XGBoost](figuras_modelo_matematico/esquema_boosting.png)

---

## 4. Red Neuronal Perceptrón Multicapa (`MLPClassifier`)

Red Neuronal Artificial densa con arquitectura feedforward y entrenamiento por retropropagación del gradiente (*Backpropagation*).

### 4.1 Propagación Hacia Adelante (Forward Propagation)
Para cada capa oculta $l \in \{1, 2\}$ con matriz de pesos $\mathbf{W}^{[l]}$ y vector de sesgos $\mathbf{b}^{[l]}$:

$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$
$$\mathbf{a}^{[l]} = g(\mathbf{z}^{[l]})$$

- **Capa 0 (Entrada):** $\mathbf{a}^{[0]} = \mathbf{x} \in \mathbb{R}^{128}$ (características preprocesadas).
- **Función de Activación Oculta (ReLU):**
  $$\text{ReLU}(z) = \max(0, z)$$
- **Capa de Salida (Neurona Única):**
  $$\hat{y} = a^{[L]} = \sigma(z^{[L]}) = \frac{1}{1 + e^{-z^{[L]}}}$$

### 4.2 Función de Pérdida con Penalización $L_2$ ($\alpha$)
$$\mathcal{L}(\mathbf{W}, \mathbf{b}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(a_i^{[L]}) + (1 - y_i) \ln(1 - a_i^{[L]}) \right] + \frac{\alpha}{2N} \sum_{l=1}^L \|\mathbf{W}^{[l]}\|_F^2$$

donde $\|\mathbf{W}\|_F$ es la norma de Frobenius.

### 4.3 Actualización de Parámetros con Optimizador Adam
Adam (*Adaptive Moment Estimation*) computa tasas de aprendizaje adaptativas basadas en momentos de primer y segundo orden:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla_\theta \mathcal{L}_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla_\theta \mathcal{L}_t)^2$$

Corrección de sesgo:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Regla de actualización:
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

![Arquitectura del Perceptrón Multicapa](figuras_modelo_matematico/esquema_mlp.png)

---

## 5. Métricas de Evaluación y Calibración de Costos

### 5.1 Métricas de Evaluación Toxicológica
Dada la matriz de confusión con Verdaderos Positivos ($TP$), Falsos Negativos ($FN$), Falsos Positivos ($FP$) y Verdaderos Negativos ($TN$):

- **Accuracy (Exactitud General):**
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

- **Recall / Sensibilidad (Cobertura de hongos venenosos - Objetivo $\to 1.0$):**
  $$\text{Recall} = \frac{TP}{TP + FN}$$

- **Precision (Precisión en hongos clasificados como venenosos):**
  $$\text{Precision} = \frac{TP}{TP + FP}$$

- **F1-Score (Media Armónica balanceada):**
  $$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 TP}{2 TP + FP + FN}$$

### 5.2 Formulación Matemática del Umbral de Riesgo Cero ($\tau^*$)
Los falsos negativos que produce un umbral $\tau$ son:
$$FN(\tau) = \sum_{i=1}^N \mathbb{I}\left( y_i = 1 \;\land\; P(Y=1 \mid \mathbf{x}_i) < \tau \right)$$

y el mayor umbral sin falsos negativos, buscado en la rejilla $\tau \in [0.01, 0.95]$ con paso $0.01$, es:
$$\tau_{\max} = \max \left\{ \tau \in (0, 0.95] \;\middle|\; FN(\tau) = 0 \right\}$$

Elegir $\tau_{\max}$ tiene dos problemas: con un único split de validación depende de qué especímenes cayeron en él, y por construcción queda en el **borde** de la región sin falsos negativos, sin margen de seguridad.

**Calibración con validación cruzada.** El umbral se calibra con una validación cruzada estratificada de $k = 5$ particiones sobre Train + Validación (51.908 registros; el test no interviene). En cada partición se entrena un pipeline nuevo, preprocesador incluido, solo con las otras $k-1$ (sin fuga de datos), y se estima la probabilidad $\hat{p}^{\,\text{oof}}_i$ de cada registro con el modelo que no lo vio al entrenar (probabilidad *fuera de partición*). Con ellas se delimita la franja sin errores, entre el comestible con mayor probabilidad ($a$) y el venenoso con menor probabilidad ($b$):

$$a = \max_{i:\, y_i = 0} \hat{p}^{\,\text{oof}}_i, \qquad b = \min_{i:\, y_i = 1} \hat{p}^{\,\text{oof}}_i$$

Si $a < b$, todo $\tau \in (a, b]$ da $FN = 0$ y $FP = 0$, y $\tau_{\max}$ coincide con $b$. El umbral desplegado es el **centro de la franja**, redondeado hacia abajo a la centésima (el lado que protege de los falsos negativos), para quedar a la misma distancia de ambos errores:

$$\tau^* = \frac{\lfloor 50\,(a + b) \rfloor}{100}$$

El cálculo está en `recalibrar_umbral_cv.py`.

**Resultado de la calibración (Random Forest, modelo ganador):** el umbral más alto con $FN = 0$ en cada partición fue $0.77,\ 0.74,\ 0.87,\ 0.76$ y $0.61$; el valor obtenido con un único split de validación ($0.67$) dejaba pasar un hongo venenoso en la quinta. Fuera de partición, $a = 0.44$ y $b = 0.61$, de modo que $\tau^* = 0.52$, con un margen de $0.08$ hacia los comestibles y $0.09$ hacia los venenosos. Con $\tau^* = 0.52$: $FN = 0$ y $FP = 0$ en las cinco particiones, en validación y en test (en test la franja va de $0.34$ a $0.67$).

### 5.3 Evidencia Empírica en el Conjunto de Prueba (Test)

Los gráficos siguientes son generados por `notebooks/03_model_training.ipynb` sobre el conjunto de prueba (datos nunca vistos durante el entrenamiento) y el script `generar_figuras_modelo.py`.

**Comparativa de los 4 modelos en test** — 3 de 4 superan la precisión del 95 % exigida; el ganador (Random Forest) se seleccionó con el criterio jerárquico $FN=0 \to F1 \to \text{ROC-AUC} \to$ tiempo de entrenamiento:

| Modelo | Accuracy | Precision | Recall (venenoso) | F1 | ROC-AUC | FN |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8670 | 0.8817 | 0.8782 | 0.8800 | 0.9363 | 619 |
| **Random Forest (ganador)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0** |
| XGBoost | 0.9940 | 0.9978 | 0.9913 | 0.9946 | 0.9999 | 44 |
| MLP Neural Network | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |

![Comparación de rendimiento de los 4 modelos en test](figuras_modelo_matematico/comparacion_modelos_test.png)

![Curvas ROC en el conjunto de prueba](figuras_modelo_matematico/curvas_roc_test.png)

![Matrices de confusión de los 4 modelos en test](figuras_modelo_matematico/matrices_confusion_test.png)

**Calibración del umbral de riesgo cero** sobre el modelo ganador, con probabilidades fuera de partición de la validación cruzada 5-fold (generada por `recalibrar_umbral_cv.py`). La franja sombreada no tiene falsos negativos ni falsos positivos, y $\tau^*$ está en su centro:

![Ajuste asimétrico de umbral del modelo ganador](figuras_modelo_matematico/ajuste_umbral_modelo_ganador.png)

**Explicabilidad global (SHAP)** — las variables morfológicas que más aportan a la clase venenosa:

![Importancia global SHAP](figuras_modelo_matematico/shap_importancia_test.png)

### 5.4 Descomposición local nativa del Random Forest (usada en la aplicación)

La aplicación explica cada predicción sin librerías externas, con las **contribuciones de camino** (método de Saabas). Sea $p_b(t)$ la proporción de venenosos en el nodo $t$ del árbol $b$, $\text{camino}_b(\mathbf{x}) = (t_0, t_1, \dots, t_L)$ los nodos que recorre $\mathbf{x}$ desde la raíz $t_0$ hasta la hoja $t_L$, y $f(t)$ la característica con la que el nodo $t$ divide los datos:

$$\phi_j^{(b)}(\mathbf{x}) = \sum_{\ell=1}^{L} \mathbb{I}\left(f(t_{\ell-1}) = j\right) \left[ p_b(t_\ell) - p_b(t_{\ell-1}) \right]$$

La suma telescópica de cada camino da $p_b(t_L) = p_b(t_0) + \sum_j \phi_j^{(b)}(\mathbf{x})$; promediando los $B$ árboles se obtiene una descomposición **exacta** de la probabilidad del bosque (§2.3):

$$\hat{P}(Y=1 \mid \mathbf{x}) = \underbrace{\frac{1}{B}\sum_{b=1}^{B} p_b(t_0)}_{\phi_0 \;=\; 0.555} + \sum_{j=1}^{d} \underbrace{\frac{1}{B}\sum_{b=1}^{B} \phi_j^{(b)}(\mathbf{x})}_{\phi_j(\mathbf{x})}$$

Las contribuciones de las columnas One-Hot de una misma variable se suman, $\Phi_k(\mathbf{x}) = \sum_{j \in k} \phi_j(\mathbf{x})$, para reportar las **20 variables morfológicas**. La **importancia global** es $I_k = \frac{1}{n}\sum_{i=1}^{n} |\Phi_k(\mathbf{x}_i)|$ sobre 300 especímenes de test, precalculada en el notebook 03 y guardada en `model_metadata.json`.

El simulador **«¿qué pasaría si…?»** evalúa $\hat{P}(Y=1 \mid \mathbf{x}_{k \leftarrow v})$ para cada valor $v$ de la variable $k$, dejando fijas las demás, y lo compara con $\tau^*$.
