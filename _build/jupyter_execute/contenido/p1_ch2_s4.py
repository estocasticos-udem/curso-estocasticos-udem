#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.stats import trim_mean


# # Reglas basicas de probabilidad

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

# # Estrategias para calcular la probabilidad de un evento
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
