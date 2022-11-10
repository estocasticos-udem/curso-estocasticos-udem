#!/usr/bin/env python
# coding: utf-8

# # Medidas
# 
# ## Estimación de localización
# 
# En análisis de datos más formal a menudo requiere el cálculo e interpretación de medidas resumidas numéricas. Es decir, de los datos se trata de extraer varios números resumidos, números que podrían servir para caracterizar el conjunto de datos y comunicar algunas de sus características prominentes.
# 
# Supongase que se tiene un conjunto de datos de la forma: $x_1,x_2,...,x_n$ donde cada $x_i$ es un dato numerico. *¿Qué características del conjunto de números son de mayor interés y merecen énfasis?* Una importante característica de un conjunto de números es su ubicación y en particular su centro. A continuación, se presentan algunos métodos para describir la ubicación de un conjunto de datos.
# 
# |Libreria | Link|
# |---|---|
# |NumPy|https://numpy.org/doc/stable/reference/routines.statistics.html|
# |Pandas|https://www.tutorialspoint.com/python_pandas/python_pandas_descriptive_statistics.htm|
# |scipy.stats|https://docs.scipy.org/doc/scipy/tutorial/stats.html|
# 
# ### Media
# 
# La medida más conocida y útil del centro es la media o promedio aritmético del conjunto.
# 
# ```{admonition} Definición
# La **media** $\bar{x}$ de un conjunto de observaciones $x_1,x_2,...,x_n$ se calcula sumando cada una de las observaciones y dividiendolas por el total de observaciones.
# 
# $$
# \bar{x}=\frac{x_1 + x_2 + ... + x_n}{n}=\frac{\sum_{n}^{i=1}x_i}{n}
# $$
# ```
# 
# Aplicando la definición anterior, si nos piden la media de los numeros: $\mathbf{x} = \left \{ 2, 3, 4, 5 \right \}$ el resultado será:
# 
# $$
# \bar{x}=\frac{\sum_{4}^{i=1}x_i}{4}=\frac{2 + 3 + 4 + 5}{4} = \frac{11}{4}=2.75
# $$
# 
# Uno de los problemas que tiene la media, es su sensibilidad a valores extremos (**outliers**). 
# 
# ### Media Truncada
# 
# Es una variación de la media que se calcula ignorando un numero fijo, en cada extremo, de valores ordenados y calculando a continuación, para los valores restantes la media.
# 
# ```{admonition} Definición
# Dado un conjunto de observaciones ordenadas $x_{(1)},x_{(2)},...,x_{(n)}$ de tal manera que $x_{(1)}$ es la mas pequeña y $x_{(p)}$ es la mas grande la formula para calcular la **media truncada** descartanto los $p$ valores mas pequeños y mas grandes esta dada por:
# 
# $$
# \textbf{media truncada}=\frac{\sum_{n-p}^{i=p+1}x_i}{n-2p}
# $$
# ```
# 
# Al descartar los valores extremos, esta metrica es mas insensible a la influencia de los valores extremos lo cual es deseable.
# 
# ### Media ponderada
# 
# En muchos casos no todos los valores tienen la misma importancia y puede ser útil asocias pesos o valores a los datos dependiendo de su relevancia para determinado estudio. En ese caso no se suman los valores uno a uno sino se halla una medida conocida como **media ponderada** la cual se define a continuación. 
# 
# ```{admonition} Definición
# Para calcular la **media ponderada** se suma el producto del cada uno de los valores  ($x_1,x_2,...,x_n$) por los respectivos pesos ( $w_1,w_2,...,w_n$) asociados a cada valor y se divide el resultado por la suma de cada uno de los pesos.
# 
# $$
# \textbf{media ponderada}=\frac{x_1 w_1 + x_2 w_2 + ... + x_n  w_n}{w_1 +  w_2 + ... + w_n}=\frac{\sum_{n}^{i=1}x_i w_i}{\sum_{n}^{i=1}w_i}
# $$
# ```
# 
# La media ponderada tambien puede ser calculada **normalizando los pesos**. Los **pesos normalizados** $w_i'$ se caracterizan por que al sumarsen todos el resultado es 1:
# 
# $$
# \sum_{n}^{i=1}w_i'=1
# $$
# 
# ```{admonition} Definición
# El **peso normalizado** $w_i'$ esta dado por la expresión:
# 
# $$
# w_i'=\frac{w_i}{\sum_{n}^{i=1}w_i}
# $$
# ```
# 
# Segun lo anterior, otra forma de expresar la **media ponderada** es por medio de la expresión:
# 
# $$
# \textbf{media ponderada}=\sum_{n}^{i=1}w_i' x_i
# $$
# 
# Un ejemplo típico de uso de esta media el para el calculo de notas de un curso cuando el valor de cada una de las evaluciones tiene diferente valor.
# 
# ### Media Geometrica
# 
# En la media aritmética se suman los valores de la variables lo cual nos indica que hay un carácter aditivo, por ejemplo cuando se suman las diferentes edades para obtener una edad promedio.
# 
# Sin embargo, hay variables que presentan variaciones acumulativas, por lo que ni la suma ni la media tienen un sentido real, por ejemplo, una rebaja del 50% sobre otra rebaja del 50% no hacen en total una rebaja del 100%, lo que alude a un caráter multiplicativo.
# 
# ```{admonition} Definición
# La **Media Geometrica** se obviente mediante la siguiente expresión sobre los datos:
# 
# $$
# \textbf{media geometrica}=\sqrt[n]{\prod_{n}^{i=1}x_i}  = \sqrt[n]{x_1 \cdot x_2 \cdot...\cdot x_n} 
# $$
# 
# También tiene una versión ponderada para la media geometrica:
# 
# $$
# \textbf{media geometrica}= \left ( \prod_{i=1}^{n}{x_i}^{\alpha_i}  \right )^{\frac{1}{\sum_{i}\alpha_i}}=\left ( x_1^{\alpha_1} \cdot x_2^{\alpha_2}\cdot...\cdot x_n^{\alpha_n}\right )^{\frac{1}{\alpha_1 + \alpha_2 + \alpha_n}}
# $$
# 
# Donde $\alpha_i$ son los pesos.
# ```
# 
# ### Mediana
# 
# La palabra **mediana** es sinónimo de **medio** y la **mediana muestral** es en realidad el valor medio una vez que se ordenan las observaciones de la más pequeña a la más grande.
# 
# ```{admonition} Definición
# La **media muestral** se obtiene ordenando primero las $n$ observaciones de la más pequeña a la más grande (con todos los valores, incluidos los  repetidos, de modo que cada observación muestral aparezca en la lista ordenada). De este modo entonces la media será:
# 
# $$
# m=\left\{\begin{matrix}
# x_{\left (\frac{n+1}{2}\right )} & \mathrm{Si\: n\: es\:impar} \\ 
# \frac{x_{\left (\frac{n}{2} \right )}+x_{\left (\frac{n}{2}+1\right )}}{2} & \mathrm{Si\: n\: es\:par}
# \end{matrix}\right.
# $$
# ```
# 
# ### Percentil
# 
# ```{admonition} Definición
# Valor tal que el $P\%$ de los valores toma este valor o un valor inferior y para el $(100 - P)\%$ el porcentaje toma este valor o un valor superior.
# ```
# 
# ### Moda
# 
# ```{admonition} Definición
# La moda representa el valor (o valores) que mas se repiten en el conjunto de datos.
# ```
# 
# ## Estimación de dispersión
# 
# El reporte de una medida de centro da sólo información parcial sobre un conjunto o distribución de datos. Diferentes muestras o poblaciones pueden tener medidas idénticas de centro y aún diferir una de otra en otras importantes maneras, una de las cuales tiene que ver en la forma como se distribuyen los datos. Las medidas principales de variabilidad (dispersión) implican las **desviaciones de la media**.
# 
# ### Rango
# 
# Es la diferencia entre los valores mayor y menor de un conjunto de datos.
# 
# ```{admonition} Definición
# Supongase que se representan los valores ordenados por $x_{(1)},x_{(2)},...,x_{(n)}$ donde x_{(1)} es el valor mas pequeño y x_{(n)} es el valor mas grande; la formula para calcular el rango esta dada por
# 
# $$
# \mathbf{rango} = x_{(n)} - x_{(1)}
# $$
# ```
# 
# ### Desviación media absoluta (Mean absolute deviation)
# 
# ```{admonition} Definición
# Las **desviaciones de la media** $d_{i}$ se obtienen restando la media \bar{x} de cada una de la observaciones muestrales $x_{1},x_{2},...,x_{n}$. Es decir:
# 
# $$
# d_{i} = x_{(i)} - \bar{x}
# $$
# ```
# 
# Dependiendo del signo de la desviación, se tienen los siguientes resultados:
# * $d_{i} > 0$: La observación es mas grande que la media.
# * $d_{i} < 0$:La observación es menor que la media.
# * $d_{i} = 0$:La observación es igual la media.
# 
# Si todas las desviaciones son pequeñas en magnitud, entonces todas las $x_i$ se aproximan a la media y hay poca variabilidad. Alternativamente, si algunas de las desviaciones son grandes en magnitud, entonces algunas $x_i$ quedan lejos de lo que sugiere una mayor cantidad de variabilidad.
# 
# Una forma simple de combinar las desviaciones en una sola cantidad es promediarlas. Desafortunadamente, esta medida no nos dira mucho pues al combinarsen las desviaciones, las desviaciones positivas compensan a las degativas por lo que la suma de estas se hace cero:
# 
# $$
# \textbf{promedio de desviaciones} = \frac{\sum_{n}^{i=1}d_i}{n} = \frac{\sum_{n}^{i=1}\left (x_i - \bar{x}\right )}{n}=0
# $$
# 
# Para evitar el problema anterior, una posibilidad es trabajar con los **valores absolutos de las desviaciones**.
# 
# ```{admonition} Definición
# La **desviación absoluta promedio** esta dada por el promedio de los valores absolutos de las desviaciones $|d_i|$
# 
# $$
# \textbf{desviación absoluta promedio} = \frac{\sum_{n}^{i=1}|d_i|}{n} = \frac{\sum_{n}^{i=1}\left |x_i - \bar{x}\right |}{n}
# $$
# ```
# 
# Como la operación de valor absoluto conduce a un número de dificultades teóricas se suele definir otras medidas mas apropiadas conocidas como la **varianza** y la **desviación estandar**.
# 
# ### Varianza 
# 
# ```{admonition} Definición
# La **varianza muestral** ($s^2$) es la suma de los cuadrados de las desviaciones de la media al cuadrado dividida por $n - 1$, donde $n$ es el numero de datos:
# 
# $$
# s^2 = \frac{\sum_{n}^{i=1}d_i^2}{n-1} = \frac{\sum_{n}^{i=1}\left (x_i - \bar{x}\right )^2}{n-1}=\frac{S_{xx}}{n-1}
# $$
# ```
# 
# ### Desviación estandar
# 
# ```{admonition} Definición
# La **desviación estandar muestral** ($s$) es la raiz cuadrada de la varianza:
# 
# $$
# s = \sqrt{s^2}
# $$
# ```
# 
# ### Resumen de los cinco números
# 
# ```{admonition} Definición
# Para un conjunto dado de números $x_1,x_2,...,x_n$, se toman las siguiente **cinco metricas**:
# 1. **Valor maximo**:
# 
# $$
# x_{min}=min(x_1,x_2,...,x_n)
# $$
# 
# 2. **Cuartil inferior**:
# 
# $$
# Q1 = P_{25}
# $$
# 
# 3. **Mediana**:
# 
# $$
# m = Q2 = P_{50}
# $$
# 
# 4. **Cuartil superior**:
# 
# $$
# Q3 = P_{75}
# $$
# 
# 5. **Valor maximo**:
# 
# $$
# x_{max}=max(x_1,x_2,...,x_n)
# $$
# 
# En resumen, los **cinco datos** estan dados por la siguiente lista:
# 
# $$
# \textbf{Min Q1 Mediana Q3 Max}
# $$
# ```
# 
# ## Rango Intercuantilico (IRQ)
# 
# ```{admonition} Definición
# El **rango intercuantilico (IRQ)** esta dado por:
# 
# $$
# IRQ = Q3 - Q1
# $$
# ```

# In[1]:


from matplotlib import rcParams, cycler
import matplotlib.pyplot as plt
import numpy as np
plt.ion()


# In[2]:


# Fixing random state for reproducibility
np.random.seed(19680801)

N = 10
data = [np.logspace(0, 1, 100) + np.random.randn(100) + ii for ii in range(N)]
data = np.array(data).T
cmap = plt.cm.coolwarm
rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, N)))


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                Line2D([0], [0], color=cmap(.5), lw=4),
                Line2D([0], [0], color=cmap(1.), lw=4)]

fig, ax = plt.subplots(figsize=(10, 5))
lines = ax.plot(data)
ax.legend(custom_lines, ['Cold', 'Medium', 'Hot']);


# There is a lot more that you can do with outputs (such as including interactive outputs)
# with your book. For more information about this, see [the Jupyter Book documentation](https://jupyterbook.org)
