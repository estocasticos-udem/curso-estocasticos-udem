#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.stats import trim_mean


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

# ## Estrategias para calcular la probabilidad de un evento
# 
# A continuación se resumen un conjunto de pasos cuando se va a abordar un problema que implica calcular probabilidades.
# 
# ```{tip}
# 1. Haga una lista de todos los eventos sencillos del espacio muestral.
# 2. Asigne una probabilidad apropiada a cada evento simple.
# 3. Determine cuáles eventos sencillos resultan en el evento de interés.
# 4. Sume las probabilidades de los eventos sencillos que resulten en el evento de interés.
# ```

# # Ejemplos usando python

# In[2]:


from fractions import Fraction

# Calculo de la probabilidad
def P(event, space): 
    "The probability of an event, given a sample space."
    return Fraction(cases(favorable(event, space)), 
                    cases(space))

favorable = set.intersection # Outcomes that are in the event and in the sample space
cases     = len              # The number of cases is the length, or size, of a set


# ## Ejemplo 1
# Se tiene un mazo de cartas imparcial y bien mezclado de 52 cartas el cual consta de cuatro palos. Los palos son tréboles (T), diamantes (D), corazones (C) y picas (P). Hay 13 cartas en cada palo que consisten en 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, J (sota), Q (reina) y K (rey) de ese palo. 
# 
# Si se saque dos cartas de un mazo estándar de 52 cartas con reemplazo. Calcule la probabilidad de obtener una carta negra como mínimo.
# 
# **Definamos los eventos**:
# * $N$: El evento de sacar una carta negra.
# * $R$: El evento de sacar una carta roja.
# * $N_i$: El evento de sacar una carta negra en el intento $i-esimo$
# * $R_i$: El evento de sacar una carta roja en el intento $i-esimo$
# 
# Lo que nos piden es:
# 
# $$
# P((N_1 \bigcap R_2) \bigcup (R_1 \bigcap N_2) \bigcup (N_1 \bigcap N_2)) = P(N_1R_2 + R_1N_2 + N_1N_2)
# $$ 
# 
# Asi: 
# 
# $$P(N_1R_2 + R_1N_2 + N_1N_2) = P(N_1R_2) + P(R_1N_2) + P(N_1N_2)$$
# 
# Como los eventos $N_1$ y $N_2$ son idenpendientes tenemos que: 
# 
# $$P(N_1R_2 + R_1N_2 + N_1N_2) = P(N_1R_2) + P(R_1N_2) + P(N_1N_2) = P(N_1)P(R_2) + P(R_1)P(N_2) + P(N_1)P(N_2)$$
# 
# Donde:
# 
# $$P(N) = \frac{N(N)}{N} = \frac{N(T \bigcup P)}{N} = \frac{13 + 13}{52} = \frac{26}{52} = \frac{1}{2}$$
# 
# $$P(R) = \frac{N(R)}{N} = \frac{N(H \bigcup D)}{N} = \frac{13 + 13}{52} = \frac{26}{52} = \frac{1}{2}$$
# 
# Ademas: $P(N) = P(N) = P(N_i) = P(R_i) = \frac{1}{2}$
# 
# $$
# P(N_1R_2 + R_1N_2 + N_1N_2) =  
#                                 \left ( \frac{1}{2} \right )\left ( \frac{1}{2} \right ) + 
#                                 \left ( \frac{1}{2} \right )\left ( \frac{1}{2} \right ) + 
#                                 \left ( \frac{1}{2} \right )\left ( \frac{1}{2} \right ) 
# $$
# 
# $$
# P(N_1R_2 + R_1N_2 + N_1N_2) =  \frac{3}{4} 
# $$
# 
# 
# Esto tambien se pudierda haber hecho así:
# 
# $$
# P(al\;menos\;una\;negra) = 1 - (ninguna\;negra) = 1 - P(ambas\;rojas) = 
# $$
# 
# $$
# P(N_1R_2 + R_1N_2 + N_1N_2) = 1 - P(R_1R_2) = 1 - P(R_1)P(R_2) = 1 - \left ( \frac{1}{2} \right )\left ( \frac{1}{2} \right ) = 1 - \frac{1}{4} = \frac{3}{4}
# $$
# 
# 

# In[3]:


# Espacio muestral
palos = set(u'♥♠♦♣')
rangos = u'K,Q,J,10,9,8,7,6,5,4,3,2,1'.split(sep = ',')
mazo  = {r + s for r in rangos for s in palos}
print("--- Mazo de cartas (Espacio muestral) ---")
print("S = ", mazo, sep = "")
print("N = ", len(mazo), sep = "")


# In[4]:


# Palos agrupados por color
p_negra = set(u'♠♣')
p_rojos = set(u'♥♠')

# Funcion que determina si una carta es negra
is_negra = lambda carta: carta[-1] in p_negra

# Funcion que retorna un conjuto de cartas negras del mazo
cartas_negras = set(filter(is_negra, mazo))

print("--- Cartas negras ---")
print("N = ", cartas_negras, sep = "")
print("N(N) = ", len(cartas_negras), sep = "")
P_N = P(cartas_negras, mazo)
print("P(N) = ",P_N , sep = "")

print("--- Cartas rojas ---")
cartas_rojas = mazo - cartas_negras
print("R = ", cartas_rojas, sep = "")
print("N(N) = ", len(cartas_rojas), sep = "")
P_R = P(cartas_rojas, mazo)
print("P(R) = ", P_R, sep = "")

"""
Probabilidad de sacar al menos una carta negra cuando se sacan dos cartas del mazo (con reemplazo)
"""
print("Calculo de la probabilidad de sacar al menos una carta negra de dos cartas seleccionas con reemplazo")

# Forma 1
P_min1_negra_f1 = P_N*P_R + P_R*P_N + P_N*P_N
print("Forma 1: P(al menos una negra) = ", P_min1_negra_f1, sep = "")

# Forma 2
P_min1_negra_f2 = 1 - P_R*P_R
print("Forma 2: P(al menos una negra) = ", P_min1_negra_f2, sep = "")


# ## Ejemplo 2
# 
# Un lote de 10000 chips de computadora utilizados en calculadoras gráficas consta de 2500 fabricados por una empresa y 7500 fabricados por una segunda empresa, todos mezclados. Si se seleccionan tres chips al azar sin reemplazo y se definen los siguintes eventos:
# * $E_1$: Evento de que el primer chip sea fabricado por la empresa 1.
# * $E_2$: Evento de que el segundo chip sea fabricado por la empresa 1.
# * $E_3$: Evento de que el tercer chip sea fabricado por la empresa 1.
# 
# **Preguntas**:
# 1. ¿Si en el proceso de selección de chips, las dos primeras elecciones correspondieron a chips de la empresa 1, ¿Cual es la probabilidad de que el tercer elegido chip pertenezca a la empresa 1?
#    
#    Para este caso nos piden: $P(E_3|E_2\;and\;E_1)$. Para resolver esto, observemos la siguiente tabla en la cual se muestra el proceso de selección: $ \left \{ E_1, E_2, E_3 \right \} = \left \{chip\;empresa\;1, chip\;empresa\;1,chip\;empresa\;1\right \}$:
#    
#    |Selección|1|2|3|
#    |--|--|--|---|
#    |Chip elegido|Empresa 1|Empresa 1| Empresa 1|
#    |Total chips|10000|9999|9998|
#    |Total chips empresa 1|2500|2499|2498|
#    |Total chips empresa 2|7500|7500|7500|
# 
#    De este modo tenemos que:
# 
#    $$
#    P(E_3|E_2\;and\;E_1)) = \frac{N(E_3 | E_2\;and\;E_1)}{N} = \frac{2498}{9998} = 0.24985
#    $$
# 
# 2. ¿Si las dos primeras elecciones correspondieron a chips de la empresa 2, ¿Cual es la probabilidad de que el tercer elegido chip pertenezca a la empresa 1?
# 
#    Lo que se nos pide en este caso es: $P(E_3|not\;E_2\;and\;not\;E_1) = P(E_3|E_2'\;and\;E_1')$
# 
#    El proceso de selección dá la siguiente secuencia: $\left \{ E_1, E_2, E_3 \right \} = \left \{chip\;empresa\;2, chip\;empresa\;2,chip\;empresa\;1\right \}$ la cual se detalla en la siguiente tabla: 
# 
#    |Selección|1|2|3|
#    |--|--|--|---|
#    |Chip elegido|Empresa 2|Empresa 2| Empresa 1|
#    |Total chips|10000|9999|9998|
#    |Total chips empresa 1|2500|2500|2500|
#    |Total chips empresa 2|7500|7499|7498|
# 
#    De este modo tenemos que:
# 
#    $$
#    P(E_3|E_2'\;and\;E_1')) = \frac{N(E_3 | E_2'\;and\;E_1')}{N} = \frac{2500}{9998}= 0.25005
#    $$

# In[5]:


# Solución python

chips = [0,0]

def init_inventario():
    chips[0],chips[1] = 2500,7500

# Actualiza el invetario dependiendo del chip elegido (Empresa 1 o 2)
def elegir_chip(empresa):
    try:
        if empresa not in [1,2]:
            raise IndexError        
    except IndexError as error:
        print("Solo exiten las empresas 1 y 2")
    else:
        chips[empresa - 1] -= 1

def seleccionar_chips(l_sel):
    i = 1
    for sel in l_sel:        
        print("{} -> Chip empresa {}".format(i,sel))
        elegir_chip(sel)
        i += 1
     
# Punto 1: seleccion = {chip empresa 1, chip empresa 1, chip empresa 1}
init_inventario()
print("Inventario antes de la selección 1:",chips)
sacadas_1 = (1,1) # El utlimo caso de seleccion 
print("Secuencia seleccionada: ")
seleccionar_chips(sacadas_1)
print("Inventario despues de la selección 1:",chips)
Prob_sel1 = float(Fraction(chips[0],sum(chips)))
print("P(E3 | E2 and E1) = {:.5f}".format(Prob_sel1))
print()
# Punto 2: seleccion = {chip empresa 2, chip empresa 2, chip empresa 1}
init_inventario()
print("Inventario antes de la selección 2:",chips)
print(chips)
sacadas_2 = (2,2)
print("Secuencia seleccionada: ")
seleccionar_chips(sacadas_2)
print("Inventario despues de la selección 2:",chips)
Prob_sel2 = float(Fraction(chips[0],sum(chips)))
print("P(E3 | E2' and E1') = {:.5f}".format(Prob_sel2))


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
