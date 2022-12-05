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
# Lo primero que se debe hacer es definir los eventos asociados al problema:
# * $R$: Evento de que la bola elegida al azar sea roja.
# * $A$: Evento de que la bola elegida al azar sea azul.
# * $B_i$: Evento de que la bola elegida provenga de la bolsa $i$, siendo $i = 1,2,3$
# 
# Para el caso lo que preguntan es cual es la probabilidad de que la bola elegida sea roja, es decir $P(R)$. Si aplicamos la ley de la probabilidad total en este caso, entonces tenemos:
# 
# $$
# P(R)= P(R\;and\;B_1) + P(R\;and\;B_2) + P(R\;and\;B_3) 
# $$
# 
# Calculando cada una de las probabilidades enteriores y teniendo en cuenta la dependencia del evento en el que se saca la bola con la bolsa de la que proviene tenemos:
# 
# $$
# P(R\;and\;B_1) = P(R|B_1)P(B_1) = \left ( \frac{75}{100} \right) \left ( \frac{100}{300} \right) = 0.25 
# $$
# 
# $$
# P(R\;and\;B_2) = P(R|B_2)P(B_2) = \left ( \frac{60}{100} \right) \left ( \frac{100}{300} \right) = 0.2 
# $$
# 
# $$
# P(R\;and\;B_3) = P(R|B_3)P(B_3) = \left ( \frac{45}{100} \right) \left ( \frac{100}{300} \right) = 0.15 
# $$
# 
# Finalmente, la probabilidad que se pide será:
# 
# $$
# P(R)= P(R\;and\;B_1) + P(R\;and\;B_2) + P(R\;and\;B_3) = 0.25 + 0.2 + 0.15 = 0.6 
# $$

# #### Ejemplo 2
# El 20% de los microprocesadores fabricados en cierto proceso son defectuosos. Si se eligen cinco microprocesadores al azar. Si se supone que funcionan de forma independiente. 
# * ¿Cuál es la probabilidad de que todos funcionen?
#   
#   Lo primero que se debe hacer es definir los eventos. De este modo si suponemos que $A$ es el evento de que un microprocesador funcione y $D$ el evento de que este defectuoso, entonces para el problema tenemos:
#   * $A_i$: Evento de que el i-esimo microprocesador elegido funcione.
#   * $D_i$: Evento de que el i-esimo microprocesador elegido no funcione (este defectuoso).
#   
#   Ahora lo que nos pregunta es $P(all\;5\;work)$, lo cual es:
# 
#   $$
#   P(all\;5\;work) = P(A_1\;and\;A_2;and\;A_3;and\;A_4;and\;A_5)
#   $$
# 
#   Ahora, como el funcionamiento de un microprocador elegido no afecta el de cualquier otro, decimos que los eventos son independientes de modo que:
# 
#   $$
#   P(A_1\;and\;A_2;and\;A_3;and\;A_4;and\;A_5) = P(A_1)P(A_2)P(A_3)P(A_4)P(A_5)
#   $$
# 
#   Del enunciado, como el 20% de los microprocesadores son defectuosos, entonces $P(D) = P(not\;A) = 0.2$ de modo que:
# 
#   $$
#   P(A) = 1- P(not\;A) = 1- P(D) = 1 - 0.2 = 0.8
#   $$
#   
#   Luego $P(A_1) = P(A_2) = P(A_3) = P(A_4) = P(A_5) = 0.8$
# 
#   Finalmente tenemos que:
# 
#   $$
#   P(A_1\;and\;A_2;and\;A_3;and\;A_4;and\;A_5) = (0.8)(0.8)(0.8)(0.8)(0.8) = 0.8^5 = 0.328
#   $$
# 
# 
# * Cual es la probabilidad de que al menos uno de los microprocesadores trabaje.
#   
#   Para solucionar este problema, podemos buscar una forma equivalente del enunciado que permita acomodar los eventos de modo que el calculo sea mas facil. En el caso, decir "que al menos uno de los microprocesadores funcione" es lo contrario de decir que "todos sean defectuosos" de modo que el punto de partida se puede expresar como se muestra a continuación:
# 
#   $$
#   P(at\;least\;one\;works) = 1 − P(all\;are\;defective)
#   $$
# 
#   Del enunciado sabems que $P(D_1) = P(D_2) = P(D_3) = P(D_4) = P(D_5) = 0.2$ y que los eventos son independientes:
# 
#   $$
#   P(all\;are\;defective) = P(D_1\;and\;D_2and\;D_3and\;D_4and\;D_5) = P(D_1)P(D_2)P(D_3)P(D_4)P(D_5)
#   $$
# 
#   $$
#   P(all\;are\;defective) = (0.2)(0.2)(0.2)(0.2)(0.2) = 0.2^5 = 0.0003
#   $$
# 
#   Finalmente: 
# 
#   $$
#   P(at\;least\;one\;works) = 1 − 0.0003 = 0.9997
#   $$
# 
#   
# 
# 

# #### Ejemplo 3
# Los clientes que compran una determinada marca de automóvil pueden solicitar un motor en cualquiera de tres tamaños disponibles. De todos los autos vendidos, el 45% tiene el motor más pequeño, el 35% tiene el mediano y el 20% tiene el más grande. De los automóviles con el motor más pequeño, el 10 % no pasa la prueba de emisiones dentro de los dos años posteriores a la compra, mientras que el 12 % de los de tamaño mediano y el 15 % de los que tienen el motor más grande fallan. ¿Cuál es la probabilidad de que un automóvil elegido al azar no pase una prueba de emisiones dentro de dos años?
# 
# Inicialmente, debemos determinar los eventos:
# * $B$: El evento de que el carro falle en la prueba de emisiones dentro de los dos años posteriores a la compra.
# * $A_1$: Evento de que el carro tenga un motor pequeño.
# * $A_2$: Evento de que el carro tenga un motor mediano.
# * $A_3$: Evento de que el carro tenga un motor grande.
# 
# Posteriormente, determinamos los datos que conocemos:
# * $P(A_1) = 0.45$
# * $P(A_2) = 0.35$
# * $P(A_3) = 0.20$
# * $P(B|A_1) = 0.10$
# * $P(B|A_2) = 0.12$
# * $P(B|A_3) = 0.15$
# 
# Finalmente, empleando la regla de la probabilidad total, calculamos la cantidad que se nos pide, la cual es $P(B)$:
# 
# $$
# P(B)= P(B\;and\;A_1) + P(B\;and\;A_2) + P(B\;and\;A_3) 
# $$
# 
# Si calculamos cada una de las probabidlidades individuales usando los datos tenemos:
# 
# $$
# P(B\;and\;A_1) = P(B|A_1)P(A_1) = (0.10)(0.45) = 0.045 
# $$
# 
# $$
# P(B\;and\;A_2) = P(B|A_2)P(A_2) = (0.12)(0.35)= 0.042 
# $$
# 
# $$
# P(B\;and\;A_3) = P(B|A_3)P(A_3) = (0.15)(0.20) = 0.03 
# $$
# 
# Finalmente tenemos:
# 
# $$
# P(B) = 0.045 + 0.042 + 0.03 = 0.117 
# $$
# 

# # Representaciones
# 
# A veces, cuando los problemas de probabilidad son complejos, puede ser útil hacer un gráfico de la situación o emplear una representación tabular. El uso de estas herramientas de representación facilita la solución de problemas que impliquen el uso de probabilidades. 
# 
# Algunas representaciones empleadas para tratar problemas de probabilidad complejos son:
# * Tablas
# * Diagramas de arbol
# * Diagramas de Venn
#   
# En las proximas secciones serán tratadas cada una de estas con mas detalle.

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
