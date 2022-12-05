#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.stats import trim_mean


# In[2]:


def prob(num_events, total):
    return num_events/total


# # Tablas de contingencia
# 
# Una **tabla de contingencia** proporciona una forma tabular de representar los datos para facilitar el cálculo de probabilidades pues facilita la determinación de condicionales con bastante facilidad. 
# 
# En esta tabla se muestra los valores de la muestra en relación con dos variables diferentes que pueden ser dependientes entre sí. 

# ## Ejemplos

# ### Ejemplo 1
# 
# La siguiente tabla de contingencia describe los 595 estudiantes que respondieron a una encuesta escolar sobre desayunar. 
# 
# ||Hombre|Mujer|Total|
# |---|---|---|---|
# |Desayuna regularmente|190|110|300|
# |No desayuna regularmente|130|165|295|
# |Total|320|275|595|
# 
# Suponga que se selecciona un estudiante al azar. Considere los eventos:
# * $B$ = desayuna regularmente 
# * $M$ = es hombre
# 
# Se pide: 
# 1. Encuentre la probabilidad de que se elija una mujer.
# 2. Encuentre la probabilidad de que la persona elegida desayune regularmente.
# 3. Encuentre la probabilidad de que la persona elegida sea una mujer que desayune regularmente.
# 4. Encuentre la probabilidad de que la persona elegida sea una mujer o cualquiera que desayune regularmente.
# 5. Encuentre $P(B \bigcup M)$ y explique lo que significa.
# 6. Encuentre $P(B^C \bigcap M^C)$ y explique lo que significa.
# 7. Encuentre $P(B|M)$ y explique lo que significa.
# 8. Encuentre $P(M|B)$ y explique lo que significa.
# 9. ¿Todos los eventos $B$ y $M$ son mutuamente excluyentes?
# 10. ¿Todos los eventos $B$ y $M$ son independientes?
# 
# #### Solución a mano
# 
# Como ya se tienen definidos los eventos básicos vamos a proceder a realizar los calculos solicitados.
# 1. $P(M') = ?$
#    
#    Tenemos que: $P(M') = 1 - P(M)$
# 
#    $$P(M) = \frac{N(M)}{N} = \frac{320}{595}$$
# 
#    Luego:
# 
#    $$P(M') = 1 - P(M) = 1 - \frac{N(M)}{N} = 1 - \frac{320}{595} = \frac{275}{595} = 0.4622$$
# 
# 2. $P(B) = ?$
# 
#    Tenemos que: 
# 
#    $$P(B) = \frac{N(B)}{N} = \frac{300}{595} = 0.5942$$
# 
# 3. $P(M'\;and\;B) = ?$
#    
#    Aplicando la definición tenemos: 
# 
#    $$P(M'\;and\;B) = \frac{N(M'\;and\;B)}{N} = \frac{110}{595} = 0.1849$$
# 
# 4. $P(M'\;or\;B) = ?$
#    
#    Tenemos que:
# 
#    $$P(M'\;or\;B) = \frac{N(M'\;or\;B)}{N}$$
# 
#    Para hallar $N(M'\;or\;B)$ es necesario tener en cuenta lo que implica la expresión "Que sea mujer o cualquier persona que desayune regularmente" es **lo contrario de decir** "Que sea hombre y no desayune regularmente" (Aplicación de ley de Morgan $(A \bigcup B)' = A' \bigcup B'$) y por lo tanto tenemos:
# 
#    $$P(M'\;or\;B) = 1 - P((M'\;or\;B)') = 1 - P(M\;and\;B')$$
# 
#    Calcular $P(M\;and\;B')$ resulta mas sencillo por lo que vamos a proceder a esto:
# 
#    $P(M\;and\;B') = \frac{N(M\;and\;B')}{N} = \frac{130}{595}$
# 
#    Luego:
# 
#    $$P(M'\;or\;B) = 1 - P(M\;and\;B') = 1 - \frac{130}{595} = \frac{465}{595} = 0.7815$$
# 
# 5. $P(B\;or\;M) = ?$
#    
#    La probabilidad $P(B\;or\;M)$ esta asociada al evento "Que sea hombre o cualquier persona que desayune regularmente". Este evento, se puede descomponer en tres eventos simples:
#    * Que sea hombre y desayune normalmente $(M\;and\;B)$.
#    * Que sea mujer y desayune normalmente $(M'\;and\;B)$.
#    * Que sea hombre y no desayune normalmente $(M\;or\;B')$.
#   
#    De este modo tenemos que:
# 
#    $$P(B\;or\;M) = \frac{N(B\;or\;M)}{N}$$
# 
#    El calculo de $N(B\;or\;M)$ se hace teniendo en cuenta los tres eventos sencillos enunciados previamente:
# 
#    $N(B\;or\;M) = N(M\;and\;B) + N(M'\;and\;B) + N(M\;and\;B') = 190 + 110 + 130 = 430$
# 
#    $N = 595$
# 
#    Luego:
# 
#    $$P(B\;or\;M) = \frac{N(B\;or\;M)}{N} = \frac{430}{595} = 0.7227$$
# 
# 6. $P(B'\;and\;M') = ?$
#    
#    La expresión $B'\;and\;M'$ significa "Que no desayune regularmente y que sea mujer":
# 
#    $$P(B'\;and\;M') = \frac{165}{595} = 0.2773$$
# 
# 7. $P(B|M) = ?$
#    
#    De la definición de probabilidad condicional tenemos:
# 
#    $$P(B|M) = \frac{N(B\;and\;M)}{N(M)} = \frac{190}{320} = 0.5938$$
# 
# 8. $P(M|B) = ?$
#    
#    De la definición de probabilidad condicional tenemos:
# 
#    $$P(M|B) = \frac{N(M\;and\;B)}{N(B)} = \frac{190}{300} = 0.6333$$
# 
# 9. No lo son, pues pueden darse de manera simultanea.
# 10. No lo son, pues hay dependencia.

# #### Solución empleando python

# In[3]:


# Representación de la tablac de datos del problema
data = {
  "regular_breakdast": np.array(["si", "no"], dtype="str"),
  "man": np.array([190, 130]),
  "woman": np.array([110, 165])
}

tabla = pd.DataFrame(data)
tabla


# In[4]:


print("--- Cantidad total de \"hombres\" y \"mujeres\" --- ")
total_sex = tabla[['man','woman']].sum(axis = 0) # Suma columnas
total_sex


# In[5]:


print("--- Cantidad total de personas que \"no desayuna\" y \"desayuna\" regularmente --- ")
total_breakfast = tabla.sum(axis = 1, numeric_only=True) # Suma Filas
total_breakfast


# In[6]:


# Punto 1
print("Punto 1 ->")
N_woman =  total_sex['woman']
N = total_sex.sum() 
P_notM = prob(N_woman,N)
print(f"P(M') = {P_notM:.4f}")


# In[7]:


# Punto 2
print("Punto 2 ->")
N_break =  total_breakfast[0]
N = total_breakfast.sum()
P_B = prob(N_break,N)
print(f"P(B) = {P_B:.4f}")


# In[8]:


# Punto 3 
print("Punto 3 ->")
N_woman_and_break =  tabla.iloc[0]['woman']
N = total_breakfast.sum()
P_notM_and_break = prob(N_woman_and_break,N)
print(f"P(M' and B) = {P_notM_and_break:.4f}")


# In[9]:


# Punto 4
print("Punto 4 ->")
N_woman_or_break =  total_sex['woman'] + total_breakfast[0] - tabla.iloc[0]['woman']
N = total_breakfast.sum()
P_notM_or_break = prob(N_woman_or_break,N)
print(f"P(M' or B) = {P_notM_or_break:.4f}")


# In[10]:


# Punto 5
print("Punto 5 ->")
N_break_or_man =  total_sex['man'] + total_breakfast[0] - tabla.iloc[0]['man'] # Mirar si se puede hacer de otra forma
                                                                               # Definicion
N = total_sex.sum()
P_B_dado_M = prob(N_break_or_man,N)
print(f"P(B or M) = {P_B_dado_M:.4f}")


# In[11]:


# Punto 6 
print("Punto 6 ->")
N_notBreak_and_woman =  tabla.iloc[1]['woman']
N = total_breakfast.sum()
P_notM_dado_notB = prob(N_notBreak_and_woman,N)
print(f"P(M' and B') = {P_notM_dado_notB:.4f}")


# In[12]:


# Punto 7
print("Punto 7 ->")
N_break_and_man =  tabla.iloc[0]['man']
N_man = total_sex['man']
P_M_dado_B = prob(N_break_and_man,N_man)
print(f"P(B|M) = {P_M_dado_B:.4f}")


# In[13]:


# Punto 8
print("Punto 8 ->")
N_break_and_man =  tabla.iloc[0]['man']
N_breakfast = total_breakfast[0]
P_M_dado_B = prob(N_break_and_man,N_breakfast)
print(f"P(M|B) = {P_M_dado_B:.4f}")


# ### Ejemplo 2
# 
# Una encuesta de 4826 adultos jóvenes (de 19 a 25 años) seleccionados al azar preguntó: "¿Cuáles crees que son las posibilidades de que tengas mucho más que un ingreso de clase media a los 30 años?" La siguiente tabla muestra las respuestas. 
# 
# |Opinion|Mujer|Hombre|Total|
# |---|---|---|---|
# |Casi ninguna | 96 | 98 | 194 |
# |Alguna posibilidad pero probablemente no | 426 | 286 | 712 |
# |Una probabilidad de 50-50 | 696 | 720 | 1416 | 
# |Gran probabilidad |663 | 758 | 1421 |
# |Practicamente segura |486 | 597 | 1083 |
# |Total |2367 | 2459 | 4826 |
# 
# 
# Si se elije un encuestado al azar:
# 1. Si la persona seleccionada es un hombre, cual es la probabilidad de que esta responda "Practicamente segura"
# 2. Si la persona seleccionada responde "Alguna posibilidad pero probablemente no" cual es la probabilidad de que la persona sea mujer
# 3. Encuentre $P(Gran\;probabilidad | Mujer)$
# 4. Encuentre $P(Gran\;probabilidad)$
# 5. Los eventos "Gran probabilidad" y "Mujer" son eventos independientes
# 
# #### Solución a mano
# 
# Primero definimos los eventos asociados al problema:
# * $M$: La persona seleccionada es Mujer.
# * $H$: La persona seleccionada es Hombre
# * $R_1$: La persona seleccionada respondió "Casi ninguna"
# * $R_2$: La persona seleccionada respondió "Alguna posibilidad pero probablemente no"
# * $R_3$: La persona seleccionada respondió "Una probabilidad de 50-50"
# * $R_4$: La persona seleccionada respondió "Gran probabilidad"
# * $R_5$: La persona seleccionada respondió "Practicamente segura"
# 
# Ahora teniendo en cuenta esto, procedamos a calcular lo que se pide:
# 1. $P(R_5|H) = ?$
# 
#    Sabemos que la formula de probabilidad condicional esta dada por:
# 
#    $$P(R_5|H) = \frac{P(R_5\;and\;H)}{P(H)}$$
# 
#    De la tabla tenemos:
# 
#    $P(R_5\;and\;H) =  \frac{N(R_5\;and\;H)}{N} =  \frac{597}{4826} = 0.1237$
# 
#    $P(H) = \frac{N(H)}{N} = \frac{2459}{4826} = 0.5095$
# 
#    Al reemplazar los valores anteriores tenemos:
# 
#    $$P(R_5|H) = \frac{P(R_5\;and\;H)}{P(H)} = \frac{\frac{597}{4826}}{\frac{2459}{4826}}$$
# 
#    $$P(R_5|H) = \frac{597}{2459} = 0.2428$$
# 
#    Otra forma de hallar lo anterior seria aplicando la definición de probabilidad condicional:
# 
#    $$P(R_5|H) = \frac{N(R_5\;and\;H)}{N(H)} = \frac{597}{2459} = 0.2428$$ 
# 
#    Como se puede ver en el procedimiento anterior, los resultados coinciden.
# 
# 2. $P(M|R_2) = ?$
#    
#    Para el caso, vamos a aplicar la definición de probabilidad condicional pues reduce la cantidad de calculos necesarios para llegar a la respuesta:
# 
#    $$P(M|R_2) = \frac{N(M\;and\;R_2)}{N(R_2)} = \frac{426}{712} = 0.5983$$
# 
# 
# 3. $P(R_4|M) = ?$
#    
#    Realizando los calculos tenemos:
# 
#    $$P(R_4|M) = \frac{N(R_4\;and\;M)}{N(M)} = \frac{663}{2367} = 0.2801$$
#    
# 4. $P(R_4) = ?$
# 
#    En este caso tenemos:
# 
#    $$P(R_4) = \frac{N(R_4)}{N} = \frac{1421}{4826} = 0.2944$$
# 
# 5. Para responder si $R_4$ y $M$ son eventos independientes se debe cumplir que $P(R_4|M) = P(R_4)$ sin embargo, al analizar las respuestas de los puntos 3 y 4 vemos que $P(R_4|M) \ne P(R_4)$ por lo que estos eventos **No son independientes**
# 
#    
#    

# #### Solución usando python

# In[14]:


# Datos
data = {
  "respuestas": np.array([1, 2, 3, 4, 5], dtype='str'),
  "mujeres": np.array([96, 426, 696, 663, 486]),
  "hombres": np.array([98, 286, 720, 758, 597])
}
tabla = pd.DataFrame(data)
tabla


# In[15]:


# Totales
total_sexo = tabla[tabla.columns[1:3]].sum()
total_sexo


# In[16]:


# Totales
total_respuestas = tabla.sum(axis = "columns", numeric_only = True)
total_respuestas


# In[17]:


# Punto 1
print("Punto 1 -> ")
N_R5_and_H = tabla.iloc[4]['hombres']
N_H = total_sexo[1]
P_R5_dado_H = prob(N_R5_and_H, N_H)
print(f"P(R5|H) = {P_R5_dado_H:0.4f}")


# In[18]:


print("Punto 2 -> ")
N_M_and_R2 = tabla.iloc[1]['mujeres']
N_R2 = total_respuestas[1]
P_M_dado_R2 = prob(N_M_and_R2, N_R2)
print(f"P(M|R2) = {P_M_dado_R2:0.4f}")


# In[19]:


print("Punto 3 -> ")
N_R4_and_M = tabla.iloc[3]['mujeres']
N_M = total_sexo[0]
P_R4_dado_M = prob(N_R4_and_M, N_M)
print(f"P(R4|M) = {P_R4_dado_M:0.4f}")


# In[20]:


print("Punto 4 -> ")
N_R4 = total_respuestas[3]
N = total_sexo.sum()
P_R4 = prob(N_R4, N)
print(f"P(R4) = {P_R4:0.4f}")


# ### Ejemplo 3
# 
# La siguiente tabla relaciona los pesos y las alturas de un grupo de personas que participan en un estudio de observación.
# 
# |Peso|Alto|Medio|Bajo|
# |---|---|---|---|
# |Obeso|18|28|14|
# |Normal|20|51|28|
# |Bajo peso|12|25|9|
# 
# Se pide:
# 1. Calcule el total de cada fila y columna
# 2. Calcule la probabilidad de que una persona elegida al azar de este grupo sea alta.
# 3. Calcule la probabilidad de que una persona elegida al azar de este grupo sea obesa y alta.
# 4. Calcule la probabilidad de que una persona elegida al azar de este grupo sea alta dado que es obesa
# 5. Calcule la probabilidad de que una persona elegida al azar de este grupo sea obesa, dado que es alta.
# 6. Calcule la probabilidad de que una persona elegida al azar de este grupo sea alta y de bajo peso.
# 7. ¿Los eventos obeso y alto son independientes?
# 

# #### Solución a mano
# 
# Inicialmente definamos los diferentes eventos simples asociados al problema:
# * $o$: Evento de que la persona elegida sea obesa.
# * $mw$: Evento de que la persona elegida tenga peso normal.
# * $lw$: Evento de que la persona elegida tenga bajo peso.
# * $hh$: Evento de que la persona elegida sea alta.
# * $mh$: Evento de que la persona elegida sea mediana.
# * $lh$: Evento de que la persona elegida sea bajita.
# 
# Ahora vamos a proceder a realizar cada uno de los calculos que se solicitan:
# 
# 1. Total de cada fila:
#    * **Total pesos**:
#      * **Obeso**: $N(o) = 18 + 28 + 14 = 60$ 
#      * **Normal**: $N(mw) = 20 + 51 + 28 = 99$ 
#      * **Bajo peso**: $N(lw) = 12 + 25 + 9 = 46 $ 
#    * **Total alturas**: 
#      * **Alto**: $N(hh) = 18 + 20 + 12 = 50$ 
#      * **Medio**: $N(mh) = 28 + 51+ 25 = 104$ 
#      * **Bajo**: $N(lh) = 14 + 28 + 9 = 51$ 
#    * **Total participantes**: $N = N(o) + N(mw) + N(lw) = 60 + 99 + 46 = 205$
# 
# 2. $P(hh) = ?$
#    
#    Empleando los calculos previamente realizados tenemos:
# 
#    $$P(hh) = \frac{N(hh)}{N} = \frac{50}{205} = 0.2439$$
# 
# 3. $P(o\;and\;hh) = ?$
# 
#    Usando la tabla y los calculos realizados en el punto 1 tenemos:
#    
#    $$P(o\;and\;hh) = \frac{N(o\;and\;hh)}{N} = \frac{18}{205} = 0.0878$$
# 
# 4. $P(hh|o) = ?$
#    
#    Sabemos que $P(hh|o) = \frac{P(hh\;and\;o)}{P(o)}$
# 
#    $P(hh\;and\;o) = \frac{N(hh\;and\;o)}{N} = \frac{18}{205}$
# 
#    $P(o) = \frac{N(o)}{N} = \frac{60}{205}$
# 
#    Luego:
# 
#    $$ 
#    P(hh|o) = \frac{P(hh\;and\;o)}{P(o)} = \frac{\frac{18}{205}}{\frac{60}{205}} = \frac{18}{60} = 0.3
#    $$
# 
# 5. $P(o|hh) = ?$
#    
#    Se procede de manera muy similar al punto anterior:
# 
#    $P(o\;and\;hh) = \frac{N(o\;and\;hh)}{N} = \frac{18}{205}$
# 
#    $P(hh) = \frac{N(hh)}{N} = \frac{50}{205}$
# 
#    Luego:
# 
#    $$
#    P(o|hh) = \frac{P(o\;and\;hh)}{P(hh)} = \frac{\frac{18}{205}}{\frac{50}{205}} = \frac{18}{50} = 0.36
#    $$
#    
# 
# 6. $P(hh\;and\;lw) = ?$
#    
#    De la tabla tenemos que:
# 
#    $$
#    P(hh\;and\;lw) = \frac{N(hh\;and\;lw)}{N} = \frac{12}{205} =  0.0585
#    $$
#    
# 7. Si comparamos $P(hh|o) = 0.3$ con $P(hh) = 0.2439$ vemos que los resultados son diferentes y por lo tanto, estos eventos no son independientes.

# #### Solución usando Python

# In[21]:


tabla = pd.DataFrame({
  "peso": np.array(["obeso", "normal", "bajo_peso"], dtype='str'),
  "alto": np.array([18, 20, 12], dtype='int'),
  "medio": np.array([28, 51, 25], dtype='int'),
  "bajo": np.array([14, 28, 9], dtype='int')
})
tabla


# In[22]:


# Punto 1 - Total en cada columna
print("Punto 1 -> Suma de las columnas")
alturas = tabla.sum(axis="rows", numeric_only=True)
alturas


# In[23]:


# Punto 1 - Total en cada fila
print("Punto 1 -> Suma de las filas (0: Obeso - 1: Normal - 2: Peso bajo)")
pesos = tabla.sum(axis="columns", numeric_only=True)
pesos


# In[24]:


# Punto 1 - Total personas
print("Punto 1 -> Total personas encuestadas ")
total = alturas.sum()
total


# In[25]:


# Obtención de la tabla de frecuencias (para facilitar los calculos)
freq_tabla = pd.concat([tabla[['peso']],tabla[['alto','medio','bajo']]/total],axis=1)
freq_tabla


# In[26]:


# Verificación de que la suma de todas las frecuencias de uno
sum_probs = freq_tabla[['alto','medio','bajo']].values.sum()
print(f"{sum_probs:.2f}")


# In[27]:


# Punto 2
print("Punto 2 -> ")
P_alta = freq_tabla[['alto']].sum()
print(f"P(hh) = {P_alta[0]:.4f}")


# In[28]:


# Punto 3
print("Punto 3 -> ")
P_obesa_and_alta = freq_tabla.iloc[0]['alto']
print(f"P(o and hh) = {P_obesa_and_alta:.4f}")


# In[29]:


# Punto 4
print("Punto 4 -> ")
P_alta_and_obesa = freq_tabla.iloc[0]['alto']
P_obesa = freq_tabla.iloc[0][1:].sum()
P_alta_dado_obesa = P_alta_and_obesa/P_obesa
print(f"P(hh|o) = {P_alta_dado_obesa:.4f}")


# In[30]:


# Punto 5
print("Punto 5 -> ")
P_obesa_and_alta = freq_tabla.iloc[0]['alto']
P_alta = freq_tabla[['alto']].sum()
P_obesa_dado_alta = P_obesa_and_alta/P_alta
print(f"P(o|hh) = {P_obesa_dado_alta[0]:.4f}")


# In[31]:


# Punto 6
print("Punto 6 -> ")
P_bajoPeso_and_alta = freq_tabla.iloc[2]['alto']
print(f"P(lw and hh) = {P_bajoPeso_and_alta:.4f}")


# # Referencias

# * https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/5-probability.ipynb
# * https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/6-statistics.ipynb
# * https://github.com/jonkrohn/ML-foundations
# * https://github.com/unpingco/Python-for-Probability-Statistics-and-Machine-Learning
# * https://github.com/Quantreo/UDEMY-STATISTIC-AND-PROBABILITY-for-quantitative-finance
# * https://github.com/ArmanBehnam/Courses
# * https://github.com/ElizaLo/Data-Science
# * https://github.com/Probability-Statistics-Jupyter-Notebook
# * https://ipython-books.github.io/154-computing-exact-probabilities-and-manipulating-random-variables/
# * https://realpython.com/python-itertools/
# * https://github.com/honi/uba-probabilidad-y-estadistica
# * https://www.cs.rpi.edu/~zaki/DMML/slides/pdf/ychap18.pdf
# * https://rpubs.com/jreigarcia/irisdataset
# * https://medium.com/analytics-vidhya/first-step-to-statistics-with-iris-data-3d29c0820c5d
# * https://www.kaggle.com/code/neha99/statistical-analysis-on-iris-dataset
# * https://www.kaggle.com/code/hassanamin/probability-and-statistics-with-python
# * https://www.geeksforgeeks.org/exploratory-data-analysis-on-iris-dataset/
# * http://www.lac.inpe.br/~rafael.santos/Docs/CAP394/WholeStory-Iris.html
# * https://courses.cs.ut.ee/MTAT.03.183/2017_spring/uploads/Main/example_submission.html
# * https://www.humanitiesdataanalysis.org/index.html
# * https://www.cs.bu.edu/fac/snyder/cs237/tutorials/LearningPython.html
# * https://risk-engineering.org/notebook/coins-dice.html
# * https://web.stanford.edu/class/archive/cs/cs109/cs109.1192/handouts/pythonForProbability.html
#   
