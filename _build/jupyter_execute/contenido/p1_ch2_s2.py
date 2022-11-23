#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.stats import trim_mean


# In[2]:


from fractions import Fraction

# Calculo de la probabilidad
def P(event, space): 
    "The probability of an event, given a sample space."
    return Fraction(cases(favorable(event, space)), 
                    cases(space))

favorable = set.intersection # Outcomes that are in the event and in the sample space
cases     = len              # The number of cases is the length, or size, of a set


# # Eventos independientes y Mutuamente excluyentes

# ## Probabilidad condicional
# 
# ```{admonition} Calculo de probabilidades condicionales
# Para encontrar la $P(A|B)$ se usa la expresión:
# 
# $$P(A|B) = \frac{P(A \bigcap B)}{P(B)}$$
# 
# Por otro lado, la probabilidad condicional $P(B|A)$ esta dada por:
# 
# $$P(B|A) = \frac{P(B \bigcap A)}{P(A)}$$
# ```
# 

# ### Ejemplo de repaso
# 
# Las pautas médicas recomiendan que un paciente hospitalizado que sufre un paro cardíaco debe recibir una desfibrilación (descarga eléctrica en el corazón) dentro de los 2 minutos. El paper **"Delayed Time to Defibrillation After In-Hospital Cardiac Arrest" (The New England Journal of Medicine [2008]: 9–17)** ([link](https://www.nejm.org/doi/pdf/10.1056/NEJMoa0706467#:~:text=N%20Engl%20J%20Med%202008%3B358%3A9%2D17.&text=Expert%20guidelines%20advocate%20defibrillation%20within,effect%20on%20survival%20are%20limited)) describe un estudio del tiempo de desfibrilación para pacientes hospitalizados en hospitales de diferentes tamaños.
# 
# Los autores examinaron los registros médicos de 6716 pacientes que sufrieron un paro cardíaco mientras estaban hospitalizados, registraron el tamaño del hospital y si la desfibrilación se produjo en 2 minutos o menos. Los datos de este estudio se resumen en la siguiente tabla:
# 
# ![tabla_infartos](p1_ch2_s2/ejemplo_prob_condicional.png)
# 
# Suponiendo que estos datos son representativos del grupo más grande de todos los pacientes hospitalizados que sufren un paro cardíaco. Suponga que se selecciona al azar un paciente hospitalizado que sufrió un paro cardíaco. Si los siguientes eventos son de interés:
# * S = evento de que el paciente seleccionado se encuentre en un hospital pequeño.
# * M = evento de que el paciente seleccionado se encuentre en un hospital de tamaño medio.
# * L = evento de que el paciente seleccionado se encuentre en un hospital grande.
# * D = evento de que el paciente seleccionado reciba desfibrilación en dos minutos o menos.
# 
# Calcular:
# 1. La probabilidad de que un paciente hospitalizado reciba una desfibrilación de manera oportuna (en dos minutos o menos).
#  
#    A partir de la tabla realizamos el calculo:
# 
#    $P(D)=\frac{4689}{6716} = 0.698$
# 
# 2. Si se selecciona un paciente que se encuentra dentro de un hospital pequeño, que tan probable es que dicho paciente reciba una desfibrilación en dos minutos o menos.
# 
#    Tenemos que: $P(D|S)=\frac{P(D \bigcap S)}{P(S)}$
# 
#    Primero se calcula la probabilidad de que el paciente hospitalizado en un hospital pequeño sufre el paro cardiaco y recibe la desfibrilación de manera oportuna:
# 
#    $P(D \bigcap S) = \frac{1124}{6716} = 0.167$
# 
#    Luego, la probabilidad de que el paciente que sufre el paro cardiado se encuentra hospitalizado en un hospital pequeño:
# 
#    $P(S) = \frac{1700}{6716}= 0.253$
# 
#    Finalmente, usando estas dos expresiones calculamos la probabilidad condicional que se pide:
# 
#    $P(D|S)=\frac{P(D \bigcap S)}{P(S)}=\frac{0.167}{0.253} = 0.660$
# 
# 3. Cual es la probabilidad de recibir la desfibrilación dentro del tiempo adecuado para el caso en el cual el paciente es elegido den un hospital mediano.
#    
#    Nos piden: $P(D|M)$
# 
#    $P(D|M)=\frac{P(D \bigcap M)}{P(M)}=\frac{\frac{2178}{6716}}{\frac{3064}{6716}} = \frac{2178}{3064} = 0.711$
# 
# 4. Cuando se selecciona el paciente de un hospital grande, ¿Cual es la probabilidad de que el paciente elegido reciba la desfibrilacion dentro del tiempo prudencial?
#    
#    En este caso se pide: $P(D|L)$
# 
#    $P(D|L)=\frac{P(D \bigcap L)}{P(L)}=\frac{\frac{1387}{6716}}{\frac{1952}{6716}} = \frac{1387}{1952} = 0.711$

# ## Tipos de eventos

# ### Eventos independientes
# 
# Dos eventos **$A$** y **$B$** son independientes si el conocimiento de que uno ha ocurrido no afecta la posibilidad de que ocurra el otro.
# 
# ```{admonition} Eventos independientes
# Si $A$ y $B$ son dos eventos independientes se cumple que: 
# 1. $P(A|B) = P(A)$
# 2. $P(B|A) = P(B)$
# 3. $P(A \bigcap B) = P(A)P(B)$
# ```
# 
# Para demostrar que dos eventos son independientes, basta con demostrar **solo una** de las condiciones anteriormente mostradas.
# 
# #### Ejemplo
# Los resultados de lanzar dos veces un dado imparcial son eventos independientes. El resultado de la primera lanzada no cambia la probabilidad del resultado de la segunda. 

# ### Eventos dependientes
# 
# Si dos eventos **No** son independientes, decimos que **son dependientes**. En estos, el conocimiento de que un evento ha ocurrido cambia la probabilidad de que otro ocurra. 
# 
# #### Ejemplo
# 
# Un ejemplo puede ser el caso de una población en la que el 0.1% de todos los individuos tiene determinada enfermedad. La presencia de la enfermedad no se puede discernir de apariencias externas, pero hay una prueba de diagnóstico disponible. Desafortunadamente, la prueba es no infalible: el 80% de los que tienen resultados positivos en la prueba en realidad tienen la enfermedad; la otro 20% que muestran resultados positivos en las pruebas son falsos positivos. 
# 
# 

# ### Eventos mutuamente excluyentes
# 
# Dos eventos **A** y **B** son **eventos mutuamente excluyentes** si no pueden ocurrir al mismo tiempo.
# 
# ```{admonition} Eventos mutuamente excluyentes
# Si $A$ y $B$ son **eventos **mutuamente excluyentes** esto siguifica que $A$ y $B$ no comparten ningún resultado y por lo tanto: 
# 
# $$
# P(A \bigcap B) = P(A)P(B)
# $$
# ```
# 
# #### Ejemplo 
# 
# Supongamos que el espacio muestral ```S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}```. Supongamos que ```A = {1, 2, 3, 4, 5}```, ```B = {4, 5, 6, 7, 8}```, y ```C = {7, 9}```. Veamos los siguientes eventos:
# * **Caso donde los eventos no son mutuamente excluyentes**: Para el caso ```A and B = {4, 5}``` por lo que ```P(A and B) = 2/10``` y no es igual a cero. Por lo tanto, ```A``` y ```B``` no son mutuamente excluyentes. 
# * **Caso donde los eventos son mutuamente excluyentes**: ```A``` y ```C``` no tienen ningún número en común por lo que ```P(A and C) = 0```. Por lo tanto, ```A``` y ```C``` son mutuamente excluyentes.
# 
# 
# ```{tip}
# A continuación se enuncian unos tips importantes:
# * Si no se sabe si $A$ y $B$ son independientes o dependientes, suponga que son dependientes hasta que pueda demostrar lo contrario.
# * Si no se sabe si $A$ y $B$ son mutuamente excluyentes, suponga que no lo son hasta que pueda demostrar lo contrario. 
# ```

# ### Ejemplos
# 
# #### Ejemplo 1
# Se tiene un experimento en el que se elige a un adulto al azar y para el cual se definen los siguientes dos eventos:
# * A = la persona tiene un nivel de colesterol de 240 miligramos por decilitro de sangre (mg/dl) o superior (colesterol alto).
# * B = la persona tiene un nivel de colesterol de 200 a 239 mg/dl (colesterol alto en el límite).
# 
# De acuerdo con la Asociación America del corazon, $P(A) = 0.16$ y $P(B) = 0.29$
# 
# 1. Explique por que los eventos $A$ y $B$ son mutuamente excluyentes.
#    
#    Los eventos A y B no se pueden dar al mismo tiempo pues una persona no pude tener a la vez el colesterol por debajo encima de 240 mg/dl y dentro del intervalo [200,239] a la vez. Por lo tanto para este caso:
# 
#    $$
#    P(A\;and\;B) = 0
#    $$
# 
# 2. Diga en palabras lo que significa el evento $A\;or\;B$. ¿Cual es la probabilidad $P(A\;or\;B)$ ? 
#   
#    $$
#    P(A\;or\;B) = P(A) + P(B) = 0.16 + 0.29 = 0.45
#    $$
# 
# 3. Si $C$ es el evento de que la persona elegida tenga un colesterol normal (por debajo de 200 mg/dl), ¿cuál es $P(C)$?
#    
#    Para el caso tenemos que: $P(C) = 1 - P(not\;C)$, luego sabemos que $P(not\;C) = P(A\;or\;B) = 0.45$ De modo que:
# 
#    $$
#    P(C) = 1 - P(not\;C) = 1 - P(A\;or\;B) = 1 - 0.45 = 0.55
#    $$
# 
# 
# 
# 
# 

# #### Ejemplo 2
# 
# Toda la sangre humana se puede clasificar en cuatro tipos distintos (O, A, B o AB); sin embargo, la distribución de los tipos varía un poco según la raza. La siguiente tabla muestra la distribución del tipo de sangre de un estadounidense negro elegido al azar:
# 
# |Tipo de Sangre|O|A|B|AB|
# |---|---|---|---|---|
# |Probabilidad|0.49|0.27|0.20|?|
# 
# 1. ¿Cual es la probabilidad de que el tipo de sangre sea AB?
#    
#    $$
#    P(AB) = 1 - P(not\;AB) = 1 - P(O\;or\;A\;or\;B)
#    $$
# 
#    Como una persona no puede tener varios tipos de sangre distinta a la vez: $P(O\;and\;A\;and\;B) = 0$ y por lo tanto:
# 
#    $$
#    P(O\;or\;A\;or\;B) = P(O) + P(A) + P(B) = 0.49 + 0.27 + 0.20 = 0.96
#    $$
# 
#    Finalmente:
# 
#    $$
#    P(AB) = 1 - P(not\;AB) = 1 - P(O\;or\;A\;or\;B) = 1 - 0.96 = 0.04
#    $$
# 
# 2. ¿Cual es la probabilidad de que una persona seleccionada no tenga sangre tipo AB?
# 
#    $$
#    P(not\;AB) = 1 - P(AB) = 1 - 0.04 = 1 - 0.04 = 0.96
#    $$
#    
# 3. María tiene sangre tipo B. Ella puede recibir con seguridad transfusiones de sangre de personas con tipo de sangre O y B. ¿Cuál es la probabilidad de que un elegido al azar americano negro puede donar sangre a María?
# 
# El donande afroamericano que puede donar sangre a Maria debe tener sangre tipo O o tipo B de modo que:
# 
# $$
# P(O\;or\;B) = P(O) + P(B) = 0.49 + 0.20 = 0.69
# $$
# 

# #### Ejemplo 3
# 
# Los estudiantes de la Universidad de New Harmony recibieron 10000 calificaciones en cursos el semestre pasado. La siguiente tabla desglosa estos grados según la escuela de la universidad que impartió el curso. Las escuelas son Artes, Ingeniería, Ciencias Sociales:
# 
# |Escuela|A|B|Menor que B|
# |---|---|---|---|
# |Artes|2142|1890|2268|
# |Ingeniería|368|432|800|
# |Ciencias Sociales|882|630|588|
# 
# Las calificaciones universitarias tienden a ser más bajas en ingeniería (E) que en artes y ciencias sociales (que incluyen Salud y Servicios Humanos). Considere los siguientes dos eventos: 
# * **E**: la calificación proviene de un curso de ingenieria.
# * **L**: la calificación es inferior a una B.
# 
# 1. Encuentre $P(L)$ e interprete esta probabilidad dentro del contexto.
#    
#    Para el caso: 
#    * $N(L) = 2268 + 800 + 588 = 3656$
#    * $N = 10000$
# 
#    De este modo:
# 
#    $$
#    P(L) = \frac{N(L)}{N} = \frac{3656}{10000} = 0.3656
#    $$
# 
#    En palabras esto significa que, el 36.56% de los estudiantes tienen una calificación por debajo de B.
# 
# 2. Encuentre $P(E|L)$ y  $P(L|E)$, ¿Cuál de estas probabilidades condicionales te dice si los estudiantes de Ingenieria de esta universidad tienden a obtener calificaciones más bajas que los estudiantes de artes y ciencias sociales? Explique.
#    
#    Para hallar $P(E|L)$:
#    * $N(E \bigcap L) = 800$
#    * $N(L) = 3656$
#    
#    Luego, al emplear la formula para $P(E|L)$ tenemos: 
# 
#    $$
#    P(E|L) = \frac{N(E \bigcap L)}{N(L)} = \frac{800}{3656} = 0.2188
#    $$
# 
#    Para calcular $P(L|E)$ tenemos:
#    * $N(L \bigcap E) = 800$
#    * $N(E) = 368 + 432 + 800= 1600$
#    
#    Finalmente, $P(L|E)$: 
# 
#    $$
#    P(L|E) = \frac{N(E \bigcap L)}{N(E)} = \frac{800}{1600} = 0.5
#    $$
# 
#    Por otro lado, la probabilidad que tiene a responder a la pregunta es $P(L|E)$ pues esta, corresponde a la probabilidad de obtener una nota por debajo de B si el estudiante es de ingenieria.
# 
# 

# ## Muestreo
# 
# El muestreo hace alusión a la técnica para la selección de una muestra a partir de una población.
# 
# ![tipos_de_muestreo](p1_ch2_s2/tipos_muestreo.png)
# 
# De acuerdo a lo que se haga con la muestra, el muestreo se puede hacer con **reemplazo** o **sin reemplazo**.

# ### Muestreo con reemplazo
# 
# ```{admonition} Muestreo con reemplazo
# Cuando se hace un muestreo con reemplazo, el individuo u objeto seleccionado seleccionado (**muestra**), se vuelve a colocar en la población antes de la siguiente selección.
# ```
# 
# En el muestreo con reemplazo, como cada miembro de una población es reemplazado después de ser elegido, entonces ese miembro tiene la posibilidad de ser elegido más de una vez.
# 
# Debido a lo anteriot, los eventos se consideran **independientes**, lo que significa que el resultado de la primera elección no cambiará las probabilidades de la segunda.
# 
# #### Ejemplo - Mazo de cartas con reemplazo
# 
# Suponga que se tiene un mazo de cartas imparcial y bien mezclado de 52 cartas. Consta de cuatro palos. Los palos son tréboles, diamantes, corazones y picas. Hay 13 cartas en cada palo que consisten en 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, J (sota), Q (reina), K (rey) de ese palo.
# 
# ![mazo_cartas](p1_ch2_s2/mazo_cartas.png)
# 
# **Muestreo con reemplazo**: Supongamos que elige **tres cartas con reemplazo**. La primera carta que elige de las 52 cartas es la **Q de picas**. Despues de elegida, se vuelve a poner esta carta en el mazo, se baraja las cartas y saca una segunda carta del mazo de 52 la cual el el **diez de tréboles**. Luego se repite el procedimiento anterior y se saca del mazo de 52 cartas una tercera carta la corresponde nuevamente a la **Q de picas**. Sus elecciones son **{Q de picas, diez de tréboles, Q de picas}**. 
# 
# Notese que en este caso es posible escoger la misma carta mas de una vez (repeticamente) siendo el caso del ejemplo, la elección de la Q de picas dos veces. 

# ### Muestreo sin reemplazo
# 
# ```{admonition} Muestreo sin reemplazo
# En el muestreo sin reemplazo, el individuo u objeto seleccionado seleccionado (**muestra**), no se vuelve a colocar en la población antes de la siguiente selección.
# ```
# 
# Como en este tipo de muestreo cada miembro de una población solo lo pueden seleccionar una vez. las probabilidades de la segunda elección se ven afectadas por el resultado de la primera, lo cual hace que los eventos sean considerados como **dependientes**.
# 
# #### Ejemplo - Mazo de cartas sin reemplazo
# 
# Suponga que se tiene un mazo de cartas imparcial y bien mezclado de 52 cartas. Consta de cuatro palos. Los palos son tréboles, diamantes, corazones y picas. Hay 13 cartas en cada palo que consisten en 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, J (sota), Q (reina), K (rey) de ese palo.
# 
# ![mazo_cartas](p1_ch2_s2/mazo_cartas.png)
# 
# **Muestreo sin reemplazo**: Supongamos que elige **tres cartas sin reemplazo**. La primera carta que saca de las 52 cartas es la **K de corazones**. Luego, se pone esta carta a un lado y saca la segunda carta de las 51 que quedan en el mazo la cual es el **tres de diamantes**. Finalmente, se pone esta carta a un lado y saca la tercera carta de las 50 restantes del mazo la cual corresponde a la **J de picas**. En resumen, sus elecciones son **{K de corazones, tres de diamantes, J de picas}**. 
# 
# Como ha escogido las cartas sin reemplazo, (a diferencia del caso anterior) no puede escoger la misma carta dos veces, lo cual hace que el evento actual de elección de cada carta sea dependientes de los eventos previos.

# In[3]:


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


# * https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/5-probability.ipynb
# * https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/6-statistics.ipynb
# * https://github.com/jonkrohn/ML-foundations
# * https://github.com/unpingco/Python-for-Probability-Statistics-and-Machine-Learning
# * https://github.com/Quantreo/UDEMY-STATISTIC-AND-PROBABILITY-for-quantitative-finance
# * https://github.com/ArmanBehnam/Courses
# * https://github.com/ElizaLo/Data-Science
# * https://github.com/Probability-Statistics-Jupyter-Notebook
# * https://ipython-books.github.io/154-computing-exact-probabilities-and-manipulating-random-variables/
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
