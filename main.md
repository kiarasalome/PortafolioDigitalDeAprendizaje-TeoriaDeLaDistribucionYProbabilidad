::::::: center
**CARRERA DE COMPUTACIÓN**\

------------------------------------------------------------------------

\
**Proyecto Integrador**\
**Unidad 3**\
**Tema:** Análisis estadístico y modelado predictivo de variables
climáticas y ambientales en la provincia de Loja (2001--2024)

------------------------------------------------------------------------

:::: minipage
::: flushleft
**Integrantes:**\
Kiara Condoy\
Javier Guarnizo\
Hector Guerrero\
Ricardo Ochoa\
Emily Salas
:::
::::

:::: minipage
::: flushright
**Docente / Tutor:**\
Ing. Cristian Narvaez\
**Asignatura:**\
Teoría de la Distribución y\
Probabilidad
:::
::::

**Ciclo:** Tercer Ciclo\
**Periodo Académico:** Abril -- Agosto 2026\
Loja -- Ecuador\
2026-07-26
:::::::

# Introducción

Cuando queremos entender un problema ambiental o tecnológico en nuestra
provincia, no podemos fijarnos en una sola causa. Por ejemplo, para
entender por qué se pierden árboles o por qué falla un sistema en Loja,
no basta con mirar solo el clima o solo la temperatura; hay muchos
factores que cambian al mismo tiempo y se influyen entre sí. Estudios
previos en el sur del Ecuador demuestran que la fragmentación y pérdida
de masa forestal responde a interacciones complejas entre factores
climáticos y presiones antropogénicas [@tapia2015deforestation].

En este trabajo analizamos datos de la provincia de Loja recopilados
entre los años 2001 y 2024. El objetivo es usar herramientas matemáticas
para conectar todas estas causas a la vez, predecir con qué frecuencia
ocurren estos problemas y entender qué tan confiables son nuestras
predicciones para ayudar a tomar mejores decisiones en la región.

# Preparación de los Datos

Usamos la información recolectada en el Proyecto Integrador sobre la
provincia de Loja (2001--2024), proveniente de repositorios satelitales
globales de cobertura arbórea [@gfw2024loja], cuya metodología se
sustenta en la detección remota multitemporal desarrollada por Hansen et
al. [@hansen2013high]. Este conjunto de datos incluye registros del
clima, la temperatura y la pérdida de vegetación con el paso de los
años. Antes de procesar los datos, realizamos las siguientes etapas de
depuración:

- **Completar datos faltantes:** Si en algún año no se registró un dato,
  usamos un cálculo intermedio para llenar ese vacío sin inventar
  información.

- **Eliminar datos raros o equivocados:** Borramos valores que
  claramente eran errores de medición (como temperaturas imposibles)
  para no confundir al sistema.

- **Normalización:** Ajustamos los números para que las variables con
  cifras muy grandes no opaquen a las que tienen cifras pequeñas.

# Análisis Predictivo Multivariado: Regresión Lineal y Diagnóstico VIF

Para evaluar cuantitativamente el impacto continuo de los predictores
climáticos sobre la masa arbórea perdida ($Y = \text{Pérdida\_Ha}$),
definimos el modelo de Regresión Lineal Múltiple mediante el método de
Mínimos Cuadrados Ordinarios (OLS):

$$\begin{equation}
    \hat{y} = \beta_0 + \beta_1 X_{\text{Temp}} + \beta_2 X_{\text{Precip}} + \beta_3 X_{\text{Hum}} + \epsilon
\end{equation}$$

Donde la estimación de coeficientes busca minimizar la suma de residuos
al cuadrado ($\sum \epsilon_i^2$).

## Diagnóstico de Multicolinealidad Estructural (VIF)

En variables climáticas observacionales, es común que la humedad y la
precipitación muestren una alta dependencia mutua. Esta
multicolinealidad genera inestabilidad en la inversión de la matriz
$(X^T X)^{-1}$, lo que dispara la varianza de los coeficientes estimados
($\hat{\beta}_j$). Para detectarla y mitigarla, calculamos el Factor de
Inflación de la Varianza (VIF) para cada regresor [@kutner2005applied]:

$$\begin{equation}
    \text{VIF}_j = \frac{1}{1 - R_j^2}
\end{equation}$$

Donde $R_j^2$ es el coeficiente de determinación obtenido al regredir la
variable $X_j$ sobre todas las demás variables independientes del
modelo. Un valor de $\text{VIF} > 5$ indica problemas de colinealidad
moderada-alta que distorsionan el supuesto *ceteris paribus*
[@kutner2005applied].

``` {.python language="Python" caption="Código para ajuste de Regresión Múltiple OLS y evaluación de VIF sobre el dataset de Loja."}
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

df_loja = pd.read_csv("loja_clima_cobertura_2001_2024.csv")
X_pred = df_loja[['Temperatura_Max', 'Precipitacion_mm', 'Humedad_Rel']]
X_ols = sm.add_constant(X_pred)
Y_ols = df_loja['Perdida_Ha']

modelo_ols = sm.OLS(Y_ols, X_ols).fit()
print(modelo_ols.summary())

vif_data = pd.DataFrame()
vif_data["Variable"] = X_ols.columns
vif_data["VIF"] = [variance_inflation_factor(X_ols.values, i) for i in range(X_ols.shape[1])]
print(vif_data)
```

# Clasificación Probabilística: Regresión Logística y Decisión de Políticas

Cuando el objetivo de ingeniería ambiental cambia de predecir la
cantidad exacta de vegetación perdida a clasificar la ocurrencia de un
**Evento Ambiental Crítico** ($1 = \text{Sí}$, $0 = \text{No}$), el
método OLS falla al predecir probabilidades fuera del rango $[0, 1]$.
Por ende, se implementa la **Regresión Logística** [@hosmer2013applied].

## Fundamento Matemático y Transformación Logit

El modelo logístico proyecta la combinación lineal de los regresores a
una escala acotada mediante la función sigmoide [@hosmer2013applied]:

$$\begin{equation}
    P(Y=1|X) = \frac{1}{1 + e^{-z}} = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \dots + \beta_k X_k)}}
\end{equation}$$

La transformación equivalente de las probabilidades a la escala lineal
de *Log-Odds* (función *logit*) permite interpretar los parámetros de
forma aditiva:

$$\begin{equation}
    \text{logit}(p) = \ln\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 X_1 + \dots + \beta_k X_k
\end{equation}$$

## Estimación por Máxima Verosimilitud (MLE) y Pseudo $R^2$

La estimación de parámetros no se realiza por mínimos cuadrados, sino
mediante la maximización de la función de verosimilitud ($L$)
[@hosmer2013applied]. La bondad de ajuste se evalúa a través del Pseudo
$R^2$ de McFadden [@mcfadden1974conditional]:

$$\begin{equation}
    R^2_{\text{McFadden}} = 1 - \frac{\ln L_{\text{modelo}}}{\ln L_{\text{nulo}}}
\end{equation}$$

``` {.python language="Python" caption="Binarización de variables climáticas, estimación por MLE y evaluación del modelo logístico."}
umbral_media = df_loja['Perdida_Ha'].mean()
df_loja['Evento_Critico'] = (df_loja['Perdida_Ha'] > umbral_media).astype(int)

Y_logit = df_loja['Evento_Critico']
X_logit = sm.add_constant(df_loja[['Temperatura_Max', 'Precipitacion_mm', 'Humedad_Rel']])

modelo_logit = sm.Logit(Y_logit, X_logit).fit()
print(modelo_logit.summary())
print(f"Pseudo R2: {modelo_logit.prsquared:.4f}")
```

## Evaluación por Matriz de Confusión y Análisis del Umbral ($Cutoff$)

La salida del modelo logístico es una probabilidad continua
$P(Y=1) \in [0,1]$. Para convertir este valor en una decisión binaria,
tradicionalmente se asume un umbral por defecto de $k = 0.50$
[@fawcett2006roc]:

$$\begin{equation}
    \hat{Y} = \begin{cases} 1 & \text{si } P(Y=1|X) \ge k \\ 0 & \text{si } P(Y=1|X) < k \end{cases}
\end{equation}$$

Sin embargo, en la gestión del territorio lojano existe una asimetría de
costos crítica:

- **Falso Negativo (Error Tipo II):** Predecir ausencia de riesgo cuando
  en realidad ocurre una catástrofe forestal.

- **Falso Positivo (Error Tipo I):** Emitir una falsa alarma.

Por esta razón, demostramos la alteración del umbral técnico
reduciéndolo de $k = 0.50$ a $k = 0.25$ para priorizar la **Sensibilidad
(Recall)** sobre la Especificidad [@fawcett2006roc].

``` {.python language="Python" caption="Evaluación comparativa de matrices de confusión variando el umbral de decisión."}
from sklearn.metrics import confusion_matrix

prob_predicha = modelo_logit.predict(X_logit)

pred_050 = (prob_predicha >= 0.50).astype(int)
cm_050 = confusion_matrix(Y_logit, pred_050)

pred_025 = (prob_predicha >= 0.25).astype(int)
cm_025 = confusion_matrix(Y_logit, pred_025)

print("Matriz k=0.50:\n", cm_050)
print("Matriz k=0.25:\n", cm_025)
```

# Fase de Clasificación Probabilística

Esta fase aborda la transición del modelado continuo al de
**clasificación probabilística**: se implementa una Regresión Logística
sobre el dataset de *Global Forest Watch* [@gfw2024loja], se demuestra
el comportamiento de la curva sigmoide, se evalúa el modelo con la
matriz de confusión, y se analiza el impacto de mover el umbral de
decisión sobre las políticas regionales [@fawcett2006roc].

## Fundamento Matemático: De la Regresión Lineal a la Curva Sigmoide

$$\begin{equation}
    P(Y=1|X) = \sigma(z) = \frac{1}{1 + e^{-z}}
\end{equation}$$

De forma equivalente, la transformación *logit* vuelve la relación
lineal e interpretable:

$$\begin{equation}
    \text{logit}(p) = \ln\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 X_1 + \dots + \beta_k X_k
\end{equation}$$

Los coeficientes $\hat{\beta}_j$ se estiman por **Máxima Verosimilitud
(MLE)** [@hosmer2013applied].

## Aplicación al Caso Regional: Predicción de Alta Pérdida de Cobertura Arbórea

Se construyó la variable binaria `High_Tree_Loss` basada en los
registros territoriales de Loja [@gfw2024loja]. Como predictores se
usaron `extent_2000_ha` y `area_ha`:

::: center
  **Variable**                  **Coef.**           **Error Est.**       **z**     **P$>|z|$**
  ----------------------- ---------------------- --------------------- ---------- -------------
  Constante ($\beta_0$)         $-7.8069$                1.334          $-5.853$      0.000
  `extent_2000_ha`         $2.496\times10^{-6}$   $1.02\times10^{-6}$   $2.449$       0.014
  `area_ha`                $3.245\times10^{-6}$   $1.15\times10^{-6}$   $2.818$       0.005
:::

::: center
Pseudo $R^2$ de McFadden $= 0.7298$ \| LLR p-valor
$= 5.95\times10^{-36}$ [@mcfadden1974conditional]
:::

## Evaluación del Modelo y Análisis de Umbrales

::: center
  **Métrica**         **Umbral $k=0.50$**   **Umbral $k=0.20$**
  ------------------ --------------------- ---------------------
  Falsos Positivos             8                    12
  Falsos Negativos            10                     1
  Sensibilidad              80.39%                98.04%
  Especificidad             94.33%                91.49%
  Exactitud                 90.63%                93.23%
:::

Dado que en Loja un Falso Negativo forestal es mucho más costoso que una
falsa alarma, **se recomienda el umbral más bajo ($k=0.20$)** como
política de decisión regional [@fawcett2006roc].

# Conclusiones y Recomendaciones

## Conclusiones

1.  La pérdida de cobertura arbórea en Loja está determinada
    principalmente por la magnitud territorial disponible, no por la
    variabilidad climática interanual: `extent_2000_ha` (cobertura base)
    y `area_ha` (área territorial) resultaron predictores significativos
    ($p=0.014$ y $p=0.005$) de un evento de Alta Pérdida, con signo
    positivo en ambos casos --- solo se pierde masa arbórea
    significativa donde originalmente existía mucha y donde hay más
    superficie disponible.

2.  El modelo logístico presenta un ajuste sólido: Pseudo $R^2$ de
    McFadden $=0.7298$ y LLR $p$-valor $\approx 5.95\times10^{-36}$, lo
    que indica que las variables estructurales explican de forma robusta
    la ocurrencia de eventos críticos, muy por encima de un modelo nulo.

3.  El umbral de decisión por defecto ($k=0.50$) no es adecuado para
    este contexto de gestión de riesgo. Aunque produce buena exactitud
    global ($90.63\%$), deja pasar $10$ Falsos Negativos de $51$ eventos
    reales de Alta Pérdida (Sensibilidad $=80.39\%$), es decir, $1$ de
    cada $5$ catástrofes forestales reales no se detecta a tiempo.

4.  Bajar el umbral a $k=0.20$ mejora drásticamente la capacidad de
    detección (Sensibilidad $=98.04\%$, solo $1$ Falso Negativo) a un
    costo moderado: los Falsos Positivos suben de $8$ a $12$, y la
    Especificidad baja ligeramente de $94.33\%$ a $91.49\%$. La
    Exactitud global incluso mejora ($93.23\%$), por lo que el
    *trade-off* es favorable en este caso.

5.  La asimetría de costos ambientales justifica priorizar Sensibilidad
    sobre Especificidad. Un Falso Negativo implica pérdida irreversible
    de bioma y costos ecológicos/sociales elevados; un Falso Positivo
    solo implica movilización preventiva de recursos, un costo menor y
    reversible.

6.  El diagnóstico VIF sobre las variables climáticas (temperatura,
    precipitación, humedad) es necesario antes de interpretar los
    coeficientes del modelo OLS, dado que humedad y precipitación
    tienden a correlacionarse, lo que puede inflar la varianza de los
    $\hat{\beta}_j$ si no se controla.

## Recomendaciones

1.  **Adoptar oficialmente $k=0.20$ como umbral de decisión** en el
    sistema de alerta temprana de pérdida forestal para la provincia de
    Loja, priorizando la detección temprana sobre la reducción de falsas
    alarmas.

2.  **Diseñar un protocolo de respuesta escalonado** para los Falsos
    Positivos adicionales ($12$ vs. $8$): dado que su costo es solo
    operativo, se recomienda que la alerta con $k=0.20$ active un
    monitoreo o verificación de campo, no necesariamente una
    intervención de máxima escala, para no saturar recursos.

3.  **Incorporar `extent_2000_ha` y `area_ha` como variables
    prioritarias** en futuros sistemas de monitoreo territorial, ya que
    son las de mayor poder predictivo identificado; priorizar el
    monitoreo de zonas con alta cobertura base y gran extensión
    territorial.

4.  **Ampliar el modelo con variables climáticas adicionales**
    (temperatura, precipitación, humedad) de forma conjunta con las
    variables estructurales, verificando previamente el VIF, para
    evaluar si mejoran el Pseudo $R^2$ ya alto ($0.7298$) o si son
    redundantes frente a las variables territoriales.

5.  **Revisar y recalibrar el umbral periódicamente** (p. ej. cada nuevo
    año de datos de *Global Forest Watch*), ya que la relación óptima
    entre sensibilidad y especificidad puede cambiar con nuevas
    dinámicas de deforestación o cambios en los costos de intervención.

6.  **Comunicar a los tomadores de decisión regionales que la elección
    del umbral es una decisión de política de gestión de riesgo, no solo
    estadística** --- debe ser validada con actores del sector ambiental
    y de gestión de riesgos, no fijada unilateralmente por el equipo
    técnico.

# Repositorios de los Integrantes

Cada integrante documentó su proceso de aprendizaje individual en el
siguiente repositorio:

- **Emily Salas:** <https://share.google/BNOZKHXxRAZHtb1ek>

- **Ricardo Ochoa:**
  <https://github.com/16-01-2005/Portafolio-de-Probabilidad/blob/main/index.md>

- **Hector Guerrero:**
  <https://github.com/Guerrero1002/Teor-a-de-la-distribuci-n.git>

- **Javier Guarnizo:**
  <https://github.com/TheJavier37/Teoria_Distribucion_Probabilidad>

- **Kiara Condoy:**
  <https://github.com/kiarasalome/PortafolioDigitalDeAprendizaje-TeoriaDeLaDistribucionYProbabilidad>
