#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.stats import trim_mean


# # Reglas basicas de probabilidad

# Al calcular la probabilidad, hay que tener en cuenta dos reglas para determinar si dos eventos son independientes o dependientes y si son mutuamente excluyentes o no:
# * La regla de multiplicación.
# * La regla de adición

# ## Regla de la multiplicación y la adición
# 
# La regla de la multiplicación establece que:
# 
# ```{admonition} Regla de la multiplicación para dos eventos
# Si $A$ y $B$ son dos eventos definidos en un espacio muestral, entonces:
# 
# $$
# P(A \bigcap B)  = P(A|B)P(B)
# $$
# 
# Para el caso en el que los eventos $A$ y $B$ son **independientes**, entonces $P(A|B) = P(A)$ de modo que la expresión queda como:
# 
# $$
# P(A \bigcap B)  = P(A)P(B)
# $$
# ```
# 
# Por otro lado la ley de la adición dice:
# 
# ```{admonition} Regla de la adición para dos eventos
# Si $A$ y $B$ son dos eventos definidos en un espacio muestral, entonces:
# 
# $$
# P(A \bigcup B)  = P(A) + P(B) - P(A \bigcap B)
# $$
# 
# Cuando los eventos $A$ y $B$ son **mutuamente excluyentes**, entonces $P(A \bigcap B) = 0$ de modo que la expresión queda como:
# 
# $$
# P(A \bigcup B)  = P(A) + P(B)
# $$
# ```

# ### Ejemplos
# 
# 
# #### Ejemplo 1
# 
# Suponga que el 60% de todos los clientes de un operador de telecomunicaciones se suscribe al servicio de Internet, el 40% se suscribe al servicio telefónico y el 25% tiene ambos tipos de servicios y realice lo que se pide a continuación
# 1. Defina los eventos de acuerdo a la información suministrada.
#    
#    De acuerdo al enunciado podemos sacar los siguientes eventos de interes de acuerdo al enunciado:
#    * **E**: evento de que el cliente seleccionado tenga servicio de internet.
#    * **F**: evento de que el cliente seleccionado tenga servicio telefonico.
# 
#    Ahora describimos en terminos de los eventos, la información que se nos da del problema:
#    *  $P(E) = 0.6$
#    *  $P(F) = 0.4$
#    *  $P(E\;and\;F) = 0.25$
# 
# 2. Si se selecciona un cliente al azar, ¿cuál es la probabilidad de que tenga al menos uno de estos dos tipos de servicio?
#    
#    Se pide $P(E\;or\;F)$ la cual se calcula la ley de la adición tenemos:
#    
#    $$
#    P(E\;or\;F) = P(E) + P(F) + P(E\;and\;F) = 0.6 + 0.4 - 0.25 = 0.75
#    $$
#    
# 3. Si se selecciona un cliente al azar, ¿cuál es la probabilidad no este inscrito en ningun servicio?
# 
#    Se pide $P(not (E\;or\;F))$ la cual podemos calcular como se muestra a continuación:
#    
#    $$
#    P(not (E\;or\;F)) = 1- P(E\;or\;F) = 1 - 0.75 = 0.25
#    $$
# 
# 4. Si se selecciona un cliente al azar, ¿cuál es la probabilidad este inscrito en exactamente un servicio?
#    
#    En terminos matematicos esto es: 
#    
#    $$P(exactamente\;uno) = P(minimo\;uno) - P(ninguno) = P(E\;or\;F) - P(E\;and\;F)$$
# 
#    $$P(exactamente\;uno) = 0.75 - 0.25 = 0.5$$
# 

# #### Ejemplo 2
# 
# En una encuesta telefónica hecha a mil adultos, a los que respondieron se les preguntó acerca del gasto de una educación universitaria y la relativa necesidad de alguna forma de ayuda financiera. Quienes respondieron fueron clasificados de acuerdo a si actualmente tenían un hijo en la universidad y si pensaban que el interes de un préstamo para casi todos los estudiantes universitarios es demasiado alta, la cantidad correcta o es muy poco. Las proporciones de quienes contestaron se muestran en la tabla de probabilidad dada a continuación:
# 
# ||Demasiado alta (A)|Correcta (B)|Muy poco (C)|
# |---|---|---|---|
# |Con hijo en universidad|.35|.08|0.01|
# |Sin hijo en la universidad|.25|.20|.11|
# 
# Si se definen los siguientes eventos:
# * A: El entrevistado piensa que el interes de un préstamo es demasiado alto
# * B: El entrevistado piensa que el interes de un préstamo tiene un valor adecuado.
# * C: El entrevistado piensa que el interes de un préstamo es muy poco.
# * D: El entrevistado tiene un hijo en la universidad.
# 
# Suponga que un entrevistado se escoge al azar de entre este grupo:
# 1. ¿Cuál es la probabilidad de que el entrevistado tenga un hijo en la universidad?
#    
#    De acuerdo a la información, tener un hijo en la universidad no depende de lo que responda de modo que:
# 
#    $$
#    P(D) = P(D\;and\;A) + P(D\;and\;B) + P(D\;and\;C) = .35 + .08 + 0.01 = 0.44
#    $$ 
# 
# 2. ¿Cuál es la probabilidad de que el entrevistado no tenga un hijo en la universidad?
#    
#    Para el caso se pide: $P(not\;D) = P(D')$ de modo que:
# 
#    $$
#    P(not\;D) = P(D) = 1 - P(D) = 1 - 0.44 = 0.56
#    $$
# 
# 3. ¿Cuál es la probabilidad de que el entrevistado tenga un hijo en la universidad o piense que la carga de un préstamo es demasiado alta?
#    
#    En este caso nos piden: $P(D\;or\;A)$, inicialmente calculamos $P(A)$:
#    
#    $$
#    P(A) = P(D\;and\;A) + P(D'\;or\;A) = .35 + .25 = 0.6
#    $$
#    
#    Luego, al aplicar la ley de la adición tenemos:
# 
#    $$
#    P(D\;or\;A) = P(D) + P(A) - P(D\;and\;A) = 0.56 + 0.44 - .35 = .69
#    $$
# 
# 4. ¿Los eventos D y A son independientes?
#    
#    Los eventos son independientes si se cumple que: $P(D\;and\;A) = P(D)P(A)$, para el caso tenemos:
#    * $P(D\;and\;A) = .69$
#    * $P(D)P(A) = (.44)(.6) = .264$
# 
#    Como $P(D\;and\;A) \neq P(D)P(A)$, entonces los eventos **no son independientes**

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

# #### Ejemplo 1
# 
# Un niño tiene tres bolsas que contienen 100 canicas cada una:
# * La bolsa 1 tiene 75 canicas rojas y 25 azules;
# * la bolsa 2 tiene 60 canicas rojas y 40 azules;
# * La bolsa 3 tiene 45 canicas rojas y 55 azules.
# 
# Si el niño elije una de las bolsas al azar y luego escoje una canica de la bolsa elegida, también al azar. ¿Cuál es la probabilidad de que la canica elegida sea roja?
# 
# Solución en: https://www.probabilitycourse.com/chapter1/1_4_2_total_probability.php

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
