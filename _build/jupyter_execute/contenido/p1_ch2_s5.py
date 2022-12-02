#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.stats import trim_mean


# In[2]:


def prob(num_events, total):
    return num_events/total


# # Representaciones

# A veces, cuando los problemas de probabilidad son complejos, puede ser útil hacer un gráfico de la situación o emplear una representación tabular. El uso de estas herramientas de representación facilita la solución de problemas que impliquen el uso de probabilidades. 

# ## Tablas de contingencia
# 
# Una **tabla de contingencia** proporciona una forma tabular de representar los datos para facilitar el cálculo de probabilidades pues facilita la determinación de condicionales con bastante facilidad. 
# 
# En esta tabla se muestra los valores de la muestra en relación con dos variables diferentes que pueden ser dependientes entre sí. 

# ### Ejemplos

# #### Ejemplo 1
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
# 1. Encuentre $P(B|M)$ y explique lo que significa.
# 2. Encuentre $P(M|B)$ y explique lo que significa.
# 3. ¿Todos los eventos $B$ y $M$ son mutuamente excluyentes?
# 4. ¿Todos los eventos $B$ y $M$ son independientes?

# In[3]:



regular_breakdast = np.array([1, 0], dtype=object)
man = np.array([190, 110], dtype=object)
woman = np.array([110, 165], dtype=object)

data = {
  "regular_breakdast": np.array([1, 0]),
  "man": np.array([190, 110]),
  "woman": np.array([130, 165])
}

cont_tab = pd.DataFrame(data)
print(cont_tab)

total_sex = cont_tab[['man','woman']].sum(axis = 0) # Suma columnas
print(total_sex)
total_breakfast = cont_tab[['man','woman']].sum(axis = 1) # Suma Filas
print(total_breakfast)
# 1
total_break =  total_breakfast[0]
num_break_and_woman = cont_tab.iloc[0]['woman']
P_B_dado_M = prob(num_break_and_woman,total_break)
print(P_B_dado_M)

# 2
num_break_and_woman =  cont_tab.iloc[0]['woman']
total_woman = total_sex[1]
P_M_dado_B = prob(num_break_and_woman,total_woman)
print(P_M_dado_B)

# 3. Yes
# 4. Verificacion por formula


# #### Ejemplo 2
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
# 3. Encuentre $P("Gran probabilidad" | Mujer)$
# 4. Encuentre $P("Gran probabilidad")$
# 5. Los eventos "Gran probabilidad" y "Mujer" son eventos independientes
# 
# * **Solución**
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
#    $$P(R_5|H) = \frac{597}{2459} = 0.2459$$
# 
#    Otra forma de hallar lo anterior seria aplicando la definición de probabilidad condicional:
# 
#    $$P(R_5|H) = \frac{N(R_5\;and\;H)}{N(H)} = \frac{597}{2459} = 0.2459$$ 
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
# 4. ¿Los eventos D y A son independientes?
#    
#    

# ## Generalización de la probabilidad total
# 
# ```{admonition} Ley de la probabilidad total
# Dados los eventos $B_1,\;,B_2,\;,B_3,\;,...,B_k$ los cuales son mutuamente exclusivos con $P(B_1)+P(B_2)+P(B_3)+...+P(B_k) = 1$, entonces para cualquier evento $E$:
# 
# $$
# P(E)  = P(E \bigcap B_1) + P(E \bigcap B_2) + ... + P(E \bigcap B_k)
# P(E)  = P(E1|B_1)P(B_1) + P(E|B_2)P(B_2) + ... + P(E|B_k)P(B_k)
# $$
# ```

# ### Ejemplos

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
# 
# Otros
# 
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
