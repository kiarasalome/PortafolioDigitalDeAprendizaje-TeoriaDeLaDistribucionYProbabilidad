En esta unidad se consolidaron conocimientos de **estadística inferencial**, **simulación**, **distribuciones de probabilidad** y **pruebas de hipótesis** mediante las actividades de **Aprendizaje Práctico Experimental (APE) 006, 007, 008 y 009**. Cada práctica permitió fortalecer tanto la comprensión teórica como la aplicación de técnicas estadísticas utilizando **Python** y un **dataset regional sobre pérdida de cobertura arbórea**.


# 📈 APE 006: Distribuciones Continuas Notables

> **Objetivo:** Comprender el comportamiento de variables continuas mediante la distribución normal y verificar si un conjunto de datos sigue dicha distribución.

## Temas abordados

- Distribución Normal
- Función de densidad de probabilidad
- Área bajo la curva
- Valores estandarizados (*Z-score*)
- Gráficos Q-Q
- Test de normalidad de Shapiro-Wilk

---

## Desarrollo de la práctica

Durante la práctica se desarrollaron funciones en **Python** para representar la distribución normal y calcular probabilidades mediante el sombreado del área bajo la curva.

Además:

- Se generaron gráficos **Q-Q**.
- Se compararon cuantiles teóricos con cuantiles observados.
- Se evaluó visualmente el cumplimiento de la normalidad.

---

## Dificultades superadas

Inicialmente existía dificultad para comprender:

- el funcionamiento de los gráficos **Q-Q**;
- el propósito del **Test de Shapiro-Wilk**;
- la relación entre ambos métodos y el comportamiento de variables continuas.

La aplicación sobre datos reales permitió entender cuándo una distribución puede considerarse normal y cuándo no.

---

## Aplicación al Dataset Regional

Se utilizó la variable continua:

```text
tc_loss_ha_2024
```

correspondiente a la pérdida de cobertura arbórea durante el año 2024.

Las actividades realizadas fueron:

1. cálculo de la media;
2. cálculo de la desviación estándar;
3. estimación de probabilidades;
4. aplicación del Test de Shapiro-Wilk.

El análisis permitió demostrar que asumir una distribución normal sin validarla estadísticamente puede conducir a conclusiones incorrectas en modelos predictivos relacionados con la deforestación.

---

# 📊 APE 007: Distribuciones Muestrales y Teorema del Límite Central (TLC)

> **Objetivo:** Analizar cómo las distribuciones muestrales convergen hacia una distribución normal mediante simulaciones.

## Temas abordados

- Teorema del Límite Central (TLC)
- Simulación Monte Carlo
- Ley de los Grandes Números
- Error estándar

---

## Desarrollo de la práctica

Se generó una población con distribución **exponencial**, altamente asimétrica.

Posteriormente:

- se extrajeron múltiples muestras aleatorias;
- se calcularon sus medias;
- se construyeron histogramas;
- se analizó el efecto del tamaño de muestra sobre el error estándar.

También se evaluaron distintos tamaños de muestra:

```text
n = 5, 10, 30, ..., 500
```

observando cómo el error estándar disminuye conforme aumenta la cantidad de datos.

---

## Dificultades superadas

El concepto de **error estándar** resultaba complejo al inicio.

Tras observar múltiples simulaciones fue posible comprender que:

- representa la precisión del estimador;
- disminuye cuando aumenta el tamaño muestral;
- mejora la confiabilidad de las estimaciones.

---

## Aplicación al Dataset Regional

Debido a la fuerte asimetría de la variable:

```text
tc_loss_ha_2024
```

se aplicó la técnica de **Bootstrapping**.

El procedimiento consistió en:

- generar **500 muestras**;
- utilizar **40 observaciones** por muestra;
- calcular la media de cada una;
- representar la distribución de dichas medias.

El histograma mostró una distribución cercana a una campana de Gauss, confirmando experimentalmente el **Teorema del Límite Central**.

---

# 📉 APE 008: Inferencia Estadística e Intervalos de Confianza

> **Objetivo:** Estimar parámetros poblacionales mediante intervalos de confianza utilizando las distribuciones Z y T de Student.

## Temas abordados

- Intervalos de confianza
- Distribución Z
- Distribución T de Student
- Nivel de confianza
- Margen de error

---

## Desarrollo de la práctica

Se calcularon los límites inferior y superior del intervalo de confianza utilizando:

- media muestral;
- desviación estándar;
- error estándar.

Además, se construyeron gráficos que mostraban la relación entre:

- nivel de confianza;
- margen de error;
- costo del muestreo.

Se evidenció el principio de **rendimientos decrecientes**, donde incrementar excesivamente el nivel de confianza aumenta considerablemente el margen de error o el tamaño requerido de la muestra.

---

## Dificultades superadas

Inicialmente resultaba difícil comprender la relación entre:

- nivel de confianza;
- margen de error.

Después de analizar diversos ejemplos se entendió por qué el **95 %** representa un equilibrio adecuado entre precisión y costo.

---

## Aplicación al Dataset Regional

Se seleccionó una muestra de:

```text
n = 25
```

regiones utilizando la variable:

```text
tc_loss_ha_2024
```

Como:

- \(n < 30\)
- la desviación poblacional era desconocida,

se empleó la **Distribución T de Student** para construir un intervalo de confianza del **95 %**.

Finalmente, mediante un gráfico de barras de error se estimó el rango donde probablemente se encuentra la verdadera media poblacional de pérdida forestal.

---

# 📑 APE 009: Pruebas de Hipótesis Paramétricas y Valor-p

> **Objetivo:** Comprender el proceso de inferencia estadística mediante pruebas de hipótesis y la interpretación correcta del valor-p.

## Temas abordados

- Hipótesis nula y alternativa
- Pruebas Z
- Pruebas T
- Valor-p
- Nivel de significancia
- Sensibilidad de las pruebas en Big Data

---

## Desarrollo de la práctica

Se realizaron pruebas paramétricas utilizando Python sobre distintos escenarios.

Además, se simuló el efecto del tamaño muestral incrementándolo desde:

```text
10 → 100 000 observaciones
```

con el propósito de demostrar que muestras extremadamente grandes pueden producir valores-p muy pequeños incluso cuando las diferencias prácticas son poco relevantes.

---

## Dificultades superadas

Uno de los principales errores conceptuales consistía en interpretar el **valor-p** como la probabilidad de que la hipótesis fuera verdadera.

Posteriormente se comprendió que el valor-p representa:

> la probabilidad de obtener datos iguales o más extremos que los observados, suponiendo que la hipótesis nula sea verdadera.

---

## Aplicación al Dataset Regional

Se construyó una nueva variable:

```text
total_loss_ha_2001_2024
```

obtenida mediante la suma de todas las pérdidas anuales de cobertura forestal.

Se planteó la hipótesis:

**Hipótesis nula**

\[
H_0:\ \mu \le 5000
\]

La prueba **T de Student** produjo un:

```text
Valor-p = 0.0000
```

Dado que:

```text
Valor-p < α = 0.05
```

se rechazó la hipótesis nula.

Como conclusión, existe evidencia estadística suficiente para afirmar que la pérdida histórica promedio de cobertura arbórea supera las **5000 hectáreas por región**, lo que evidencia un problema significativo de deforestación.

---

# Conclusión General

Las prácticas APE permitieron fortalecer competencias en:

- Distribuciones de probabilidad.
- Simulación estadística.
- Teorema del Límite Central.
- Intervalos de confianza.
- Inferencia estadística.
- Pruebas de hipótesis.
- Interpretación del valor-p.

Más allá de la teoría, todas las técnicas fueron implementadas en **Python** y aplicadas a un **dataset real de deforestación**, permitiendo comprender la utilidad de la estadística como herramienta para la toma de decisiones fundamentadas en datos.
````
