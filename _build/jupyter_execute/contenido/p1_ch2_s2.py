#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.stats import trim_mean


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
# * **Valor maximo**:
# 
# $$
# x_{min}=min(x_1,x_2,...,x_n)
# $$
# 
# * **Cuartil inferior**:
# 
# $$
# Q1 = P_{25}
# $$
# 
# * **Mediana**:
# 
# $$
# m = Q2 = P_{50}
# $$
# 
# * **Cuartil superior**:
# 
# $$
# Q3 = P_{75}
# $$
# 
# * **Valor maximo**:
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

# ## Ejemplos

# ### Ejemplo 1
# 
# Asuma que la altura (en cm) de los estudiantes de una clase es como sigue: 90,102,110,115,85,90,100,110,110. ¿Cual es el promedio de alturas?

# #### Solución usando código Python

# In[2]:


# Implementacion de la media
def media(data):
    return sum(data)/len(data)

# Test
heights = [90,102,110,115,85,90,100,110,110]
mean_height = media(heights)
print("Alturas (cm):", heights)
print("Promedio de las alturas (cm):", mean_height)


# #### Solución usando numpy

# In[3]:


# Solucion numpy
heights =  np.array([90,102,110,115,85,90,100,110,110])
print("Alturas (cm):",heights)
mean_height = np.mean(heights)
print("Altura promedio (cm): {0:.2f}".format(mean_height))
#### Solución usando pandas


# #### Solución usando pandas

# In[4]:


# Solucion pandas
data_heights = {'heights':pd.Series([90,102,110,115,85,90,100,110,110])}
df_heights= pd.DataFrame(data_heights)
print(df_heights)
mean_height = df_heights.mean()
print("Altura promedio: {0:.2f}".format(mean_height[0]))


# ### Ejemplo 2
# 
# Dado los los siguientes datos: 22, 25, 29, 11, 14, 18, 13, 13, 17, 11, 8, 8, 7, 12, 15, 6, 8, 7, 9, 12. Calcule la media recortando un 10% de los datos.

# In[5]:


# Se uso la funcion trim_mean de scipy.stats

data = [22, 25, 29, 11, 14, 18, 13, 13, 17, 11, 8, 8, 7, 12, 15, 6, 8, 7, 9, 12]

#calculate 10% trimmed mean
print("Datos:",data)
print("Media 10% truncada: ",trim_mean(data, 0.1), sep = "")


# ### Ejemplo 3
# Asuma que la altura (en cm) de los estudiantes de una clase es como sigue: 90,102,110,115,85,90,100,110,110. ¿Cual es la media de alturas?

# #### Solución usando numpy

# In[6]:


heights =  np.array([90,102,110,115,85,90,100,110,110])
print("heights:",heights)
print("heights ordenadas:",np.sort(heights))

median_height = np.median(heights)
print("Mediana: {0:.2f}".format(median_height))


# #### Solución usando pandas

# In[7]:


# Solucion pandas
data_heights = {'heights':pd.Series([90,102,110,115,85,90,100,110,110])}
df_heights= pd.DataFrame(data_heights)
print(df_heights)
median_height = df_heights.median()
print("Altura promedio: {0:.2f}".format(median_height[0]))


# ### Ejemplo 4
# Asuma que la altura (en cm) de los estudiantes de una clase es como sigue: 90,102,110,115,85,90,100,110,110. ¿Cual es el rango de las alturas?

# #### Solución usando numpy

# In[8]:


# Solucion numpy
heights =  np.array([90,102,110,115,85,90,100,110,110])
print("heights:",heights)
min_height = np.min(heights)
max_height = np.max(heights)
range_height = max_height - min_height
print("Rango [{0:d},{1:d}]: {2:d}".format(min_height, max_height, range_height))


# #### Solución usando pandas

# In[9]:


# Solucion pandas
data_heights = {'heights':pd.Series([90,102,110,115,85,90,100,110,110])}
df_heights= pd.DataFrame(data_heights)
print(df_heights)
min_height = df_heights.min()
max_height = df_heights.max()
range_height = max_height[0] - min_height[0]
print("Rango [{0:d},{1:d}]: {2:d}".format(min_height[0], max_height[0], range_height))


# ### Ejemplo 5
# 
# El sitio web www.fueleconomy.gov contiene una gran cantidad de información acerca de las características del combustible de varios vehículos. Además de las calificaciones de millaje de la EPA, hay muchos vehículos para los que los usuarios han informado de sus propios valores de eficiencia de combustible (mpg). Considere la siguiente muestra de $n = 11$ eficiencias para el Ford Focus 2009 equipado con transmisión automática (para este modelo, la EPA informa de una calificación general de 27 mpg-24 mpg en ciudad y 33 mpg en carretera) las cuales se muestran a continuación:
# 
# 
# |$x_i$|
# |---|
# |27.3|
# |27.9|
# |32.9|
# |35.2|
# |44.9|
# |39.9|
# |30.0|
# |29.7|
# |28.5|
# |32.0|
#     
# Obtenga:
# 1. La media.
# 2. Las desviaciones.
# 3. Las desviaciones al cuadrado
# 4. La desviacion estandar
# 5. La varianza

# #### Solución usando numpy

# In[10]:


# Solucion numpy
# Calculos empleando funciones de acuerdo a las formulas
epa_data =  np.array([27.3, 27.9, 32.9, 35.2, 44.9, 39.9, 30.0, 29.7, 28.5, 32.0, 37.6])
epa_mean = np.mean(epa_data)
epa_desv = epa_data - np.array(epa_data.size*[epa_mean])
epa_ec = epa_desv**2
epa_var = np.sum(epa_ec)/(epa_data.size - 1)
epa_std = (epa_var)**0.5
# Despliegue de valores
print("RESUMEN DE LAS MEDIDAS DE DISPERSION")
print("-----------------------------------------------------------")
print("EPA (mpg): ",epa_data)
print("EPA Media: ",epa_mean)
print("Desviaciones EPA: ",epa_desv)
print("Error cuadratico EPA: ",epa_ec)
print("Varianza: ",epa_var)
print("Desviacion estandar EPA: ",epa_std)
print("-----------------------------------------------------------")
# Obtencion de la media, la desviacion estandar y la varianza usando las formulas de numpy
epa_mean = np.mean(epa_data)
epa_std = np.std(epa_data, ddof=1)
epa_var = np.var(epa_data, ddof=1)
print("EPA Media: ",epa_mean)
print("Varianza EPA [np.std]: ",epa_var)
print("Desviacion estandar [np.var]: ",epa_std)


# #### Solución usando pandas

# In[11]:


# Solución pandas
data = {'epa_data':pd.Series([27.3, 27.9, 32.9, 35.2, 44.9, 39.9, 30.0, 29.7, 28.5, 32.0, 37.6])}
epa_data = pd.DataFrame(data)
epa_mean = epa_data.mean()
epa_desv = epa_data - epa_mean
epa_desv = epa_desv.rename(columns={'epa_data':'epa_dist'})
epa_ec = epa_desv**2
epa_ec = epa_ec.rename(columns={'epa_data':'epa_ec'})
epa_var =  epa_data.var()
epa_std = epa_data.std()
df_summary = pd.concat([epa_data,epa_desv,epa_ec],axis=1)
df_summary


# In[12]:


print("EPA Media: ",epa_mean)
print("Varianza EPA: ",epa_var)
print("Desviacion estandar: ",epa_std)


# ### Ejemplo 6
# 
# Se utilizó ultrasonido para reunir los datos adjuntos de corrosión en el espesor de la placa de piso de un tanque elevado utilizado para almacenar petróleo crudo (“Statistical Analysis of UT Corrosion Data from Floor Plates of a Crude Oil Aboveground Storage Tank”, Materials Eval., 1994: 846–849); cada observación es la profundidad de la picadura más grande en la placa, expresada en milésimas de pulgada: 40, 52, 55, 60, 70, 75, 85, 85, 90, 90, 92, 94, 94, 95, 98, 100, 115, 125, 125.
# 
# Muestre el resumen de los cinco numeros

# #### Solución usando numpy

# In[13]:


# Usando numpy
profundidad = np.array([40, 52, 55, 60, 70, 75, 85, 85, 90, 90, 92, 94, 94, 95, 98, 100, 115, 125, 125])
min_prof = np.quantile(profundidad,q = 0.0) # np.percentile(profundidad,q = 0)
max_prof = np.quantile(profundidad,q = 1.0) # np.percentile(profundidad,q = 100)
Q1_prof = np.quantile(profundidad,q = 0.25) # np.percentile(profundidad,q = 25)
Q3_prof = np.quantile(profundidad,q = 0.75) # np.percentile(profundidad,q = 75)
median_prof = np.quantile(profundidad,q = 0.5) # np.median(profundidad)
rango_prof = np.ptp(profundidad)
print("Resumen profundidades")
print("- Minimo: ", min_prof, sep="")
print("- Q1: ", Q1_prof , sep="")
print("- Mediana: ", median_prof , sep="")
print("- Q3: ", Q3_prof , sep="")
print("- Maximo: ", max_prof, sep="")
print("- IRQ: ", Q3_prof - Q1_prof, sep="")


# #### Solución usando Pandas

# In[14]:


# Solucion pandas
data = {'epa_data':pd.Series([27.3, 27.9, 32.9, 35.2, 44.9, 39.9, 30.0, 29.7, 28.5, 32.0, 37.6])}
epa_data = pd.DataFrame(data)
epa_data.describe()

