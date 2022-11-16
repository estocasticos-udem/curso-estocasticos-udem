#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.stats import trim_mean


# # Experimentos y eventos
# 
# ## Experimento
# 
# ```{admonition} Definición
# Un **experimento** es una operación planificada que se realiza en condiciones controladas. Si el **resultado** (producto del experimento) no está predeterminado, se dice que el experimento es **aleatorio**.
# ```
# 
# Un experimento consiste en un procedimiento y unas observaciones. Por ejemplo un experimento podria ser lanzar una moneda no cargada ([simulacion](http://digitalfirst.bfwpub.com/stats_applet/stats_applet_10_prob.html)). Esto implica:
# * **Procedimiento**: Lanzar una moneda varias veces e ir anotando los resultados en una tabla.
# * **Observaciones**: Observar el lado de la moneda (cara o sello) despues de que la moneda es lanzada.
# 
# La siguiente tabla muestra algunos ejemplos de experimentos:
# 
# |Experimento|Procedimiento|Observaciones|
# |:---:|---|---|
# |**Lanzar una moneda tres veces**|Lanzar una moneda varias veces e ir apuntando los resultados en una tabla|Posibles observaciones: <ol> <li>Secuencia de caras y sellos</li> <li>Numero de caras</li> </ol>|
# |**Lanzar una par de monedas**|Lanzar un par de monedas varias veces e ir anotando los resultados en una tabla|Salida (cara/sello) de cada moneda|
# |**Contar el numero de estudiantes que asisten a clase**|Llamar a lista cada clase|La cantidad de estudiantes que asisten|
# |**Determinar la calidad del suerño**|Medir la hora de acostada y levantada y apuntarlas en una tabla|Posibles observaciones: <ol> <li>Tiempo total de dormida</li> <li>Numero veces que se desperto durante la noche</li> <li>Como se sintió con la dormida (Valor categorico en la escala 1 a 5)</li> </ol>|
# 
# ## Espacio muestral
# 
# Antes de definir el concepto de espacio muestral, es conveniente definir el concepto de salida:
# 
# ```{admonition} Salida (Outcome)
# Una **salida** (outcome) de un experimento es una de las posibles observaciones de dicho experimento.
# ```
# 
# Por ejemplo si un experimento consiste en **lanzar un dado no cargado**, la salida consiste en uno de los posibles valores que cae en la cara.
# 
# ```{admonition} Espacio muestral
# El **espacio muestral** ($S$) de un *experimento* es el conjunto de todos los resultados posibles. 
# ```
# 
# Existen 3 formas de representar un espacio muestral:
# * Hacer una lista de posibles resultados.
# * Crear un diagrama de árbol.
# * Crear un diagrama de Venn. 
# 
# ### Ejemplo 1
# 
# Considere un **experimento** para investigar si es más probable que los hombres o las mujeres elijan un auto electrico en lugar de uno a gasolina al comprar un Honda Civic en un concesionario de automóviles en particular. El Honda Civic está disponible en ambas variantes. En este experimento, se seleccionará al azar un cliente entre los que compraron un Honda Civic. Se determinará el tipo de vehículo adquirido (electrico o a gasolina) y se registrará el sexo del cliente. 
# 
# Antes de el cliente es seleccionado, el resultado de este experimento casual es desconocido para nosotros. Sin embargo, sabemos cuáles son los posibles resultados (**espacio muestral**). 
# 
# Por ejemplo, si los resultados se representan como una lista tenemos:
# 1. Mujer compra auto electrico.
# 2. Mujer compra auto a gasoluna.
# 3. Hombre compra auto electrico.
# 4. Hombre compra auto a gasoluna.
# 
# Una forma mas resulmida consiste en representar las salidas anteriores usando pares ordenados de modo que:
# 
# ```
# S = {(hombre, electrico), (mujer, electrico),
#      (hombre, gasolina), (mujer, gasolina)}
# ```
# 
# En la representación en la que se usa el diagrama de arbol para representar espacio muestral para identificar un resultado especifico (en este ejemplo), se recorre el arbol seleccionando primero la rama correspondiente al sexo (Masculino/Femenino) del comprador y luego la rama correspondinete al tipo de vehiculo (Electrico/Gasolina). La siguiente. En la siguiente figura se resalta el caso en el que la salida corresponde a la adquisición de un auto electrico por parte de un hombre:
# 
# ![diagrama_arbol](p1_ch2_s1/diagrama_arbol_carros1.png)
# 
# Como en este caso, en la elección del automovil no importa el orden de la selección, no importa cual rama va primero y cual despues de modo que una representación equivalente se muestra a continuacuón:
# 
# ![diagrama_arbol](p1_ch2_s1/diagrama_arbol_carros2.png)
# 
# ## Eventos
# 
# ```{admonition} Evento
# Un **evento** es cualquier combinación de resultados del espacio muestral asociado a un experimento aleatorio.
# ```
# 
# Cuando el resultado consisten en exactamente una sola salida, decimos que el evento es **simple**.
# 
# Usualmente se emplean letras mayusculas $(A, B, C,...)$  o letras con subindices $(E_1, E_2, E_3,...)$ para representar eventos. 
# 
# ### Ejemplo 2
# 
# Teniendo en cuenta el experimento analizado en el ejemplo 1 (adquisión del onda civic). Podemos definir los siguientes eventos:
# * **M**: El auto fue adquirido por un hombre.
# * **F**: El auto fue comprado por una mujer.
# * **G**: El tipo de carro es de gasolina.
# * **E**: El tipo de carro es electrico.
# 
# Se pide:
# 1. ¿Cual es el espacio muestral teniendo en cuenta esta representación?
# 
#    $S = \left \{ME, FE, MG, FG \right \}$
# 
# 2. Encuentre los eventos simples de cada caso.
#    *  $E_1 = ME$
#    *  $E_2 = FE$
#    *  $E_3 = MG$
#    *  $E_4 = FG$
# 
# 3. Suponiendo que el evento de interes consiste en todas las salidas cuando un auto electrico es elegido tenemos:
#    
#    $$electrico = \left \{ME, FE \right \}$$
# 
# 4. Evento en el que el comprador elegido es mujer:
#    
#    $$mujer = \left \{FE, FG \right \}$$
# 
# ### Ejemplo 3
# 
# Un experimento consiste en lanzar una moneda no cargada. ¿Cual es el espacio muestral asociado al experimento?
# 
# ![s_moneda](./p1_ch2_s1/espacio_muestral_moneda.png)
# 
# Inicialmente definimos los siguientes eventos:
# * **H**: el resultado de lanzar la moneda es cara.
# * **T**: el resultado de lanzar la moneda es sello.
# 
# De este modo, el espacio muestral del experimento esta dado por:
# 
# $$S = \left \{H, T \right \}$$
# 
# ### Ejemplo 4
# 
# Suponga que se lleva a cabo un experimento que consiste en lanzar dos veces una moneda imparcial. Si **H**, es el evento en el cual el resultado es cara y **T** es el evento cuyo resultado es sello.
# 1. ¿Cual es el espacio muestral?
# 
#    La siguiente figura muestra las difentes posibilidades a la salida:
# 
#    ![s_par_,monedas](p1_ch2_s1/espacio_muestral_par_monedas.png)
# 
#    De este modo, segun lo anterior, el espacio muestral para este experimento es:
# 
#    $$S = \left \{HH, HT, TH, TT \right \}$$
# 
# 2. Llenar la siguiente tabla:
#    
#    |Evento|Descripción|Resultado|
#    |------|------|------|
#    |$E_1$ |Que salga al menos un sello||
#    |$E_2$ |Que las salidas sean las mismas||
#    |$E_3$|Que la primera moneda sea cara||
# 
#    A continuación se muestra la tabla llena:
#    
#    |Evento|Descripción|Resultado|
#    |------|------|------|
#    |$E_1$ |Que salga al menos un sello|$E_1 = \left \{TH, HT, TT \right \}$|
#    |$E_2$ |Que las salidas sean las mismas|$E_2 = \left \{HH, TT \right \}$|
#    |$E_3$|Que la primera moneda sea cara|$E_3 = \left \{HH, HT \right \}$|
# 
# 
# Es posible crear nuevos eventos a partir de eventos ya especificados tal y como se muestra a continuación:
# 
# ```{admonition} Operaciones sobre eventos
# Dados dos eventos $A$ y $B$:
# * **Not $A$**: Evento que contiene todas las salidas del experimento que no estan en el evento $A$. **Not $A$** es algunas veces llamado **Complemento de $A$** y usualmente es denotado como: $A^c$, $A'$ o $\bar{A}$.
# * **$A$ or $B$**: Evento que consiste de todas las salidas del experimento que estan al menos en uno de los dos eventos, esto es, que estan en $A$, en $B$ o en ambos. **$A$ or $B$** es llamado la unión de los dos eventos y es denotado por $A \bigcup B$.
# * **$A$ and $B$**: Evento que consiste en todas las salidas que se encuentran en ambos eventos $A$ y $B$. **$A$ and $B$** es conocido como la intersección de los dos eventos y se denota por $A \bigcap B$.
# ```
# 
# ### Ejemplo 5
# 
# Se le ha pedido a una ingeniera de tránsito que considere si una señal de alto en la parte inferior de la rampa de salida de una autopista debe ser reemplazada por un semáforo. Para ayudar en esta decisión, ella planea observar los patrones de tráfico de esta rampa de salida. Para esto, la ingeniera registra la direccion de giro (**L**: Izquierda; **R**: Derecha) de tres vehiculos sucesivos. Para esto se pide lo siguiente:
# 1. ¿Cual es el espacio muestral del experimento?
#    
#    Sea $S$ el espacio muestral del experimento tenemos que:
# 
#    $$S = \left \{LLL, RLL, LRL, LLR, RRL, RLR, LRR, RRR \right \}$$
# 
# 2. ¿Cual es la salida para el evento de que exactamente solo un carro gire a la derecha?
#    
#    Sea $A$ = Evento de que exactamente un carro gire a la derecha, tenemos que:
# 
#    $$A = \left \{RLL, LRL, LLR \right \}$$
# 
# 3. ¿Cual es la salida del evento de que a lo sumo un carro gire a la derecha?
#    
#    Sea $B$ = Evento de que maximo un carro gire a la derecha, tenemos que:
# 
#    $$B = \left \{RLL, LRL, LLR, LLL  \right \}$$
# 
# 4. ¿Cual es la salida evento en el cual todos los carros giran en la misma dirección?
# 
#    Sea $C$ = Evento en el que todos los carros giran en la misma dirección
# 
#    $$C = \left \{RRR, LLL \right \}$$
# 
# 5. ¿Cual es la salida asociada al evento en el cual todos los carros no giran en la misma dirección?
# 
#    Sea $D$ = Evento en el que todos los carros no giran en la misma dirección
# 
#    $$D = C^C = \left \{RLL, LRL, LLR, RRL, RLR, LRR\right \}$$
# 
# 6. ¿Cual es la salida para el evento en el cual solo uno de los carros gira a la derecha o todos los carros giran en la misma dirección?
#    
#    Sea $E$ = Evento en el que solo uno de los carros gira a la derecha o todos los carros giran en la misma dirección.
# 
#    $$E = A \bigcup C  = \left \{RLL, LRL, LLR \right \} \bigcup \left \{LLL, RRR \right \} = \left \{RLL, LRL, LLR, LLL, RRR \right \}$$
# 
# 7. ¿Cual es la salida del evento de que a lo sumo (como maximo) un carro gire a la derecha y todos los autos giran en la misma dirección?
# 
#    Sea $F$ = Evento en el que como maximo un solo carro gira a la derecha y todos los autos giran en la misma dirección
# 
#    $$F = B \bigcap C = \left \{RLL, LRL, LLR, LLL \right \} \bigcap \left \{RRR, LLL \right \} = \left \{LLL\right \}$$
# 
# ## Simulación
# 
# Poner un ejemplo...

# # Probabilidades
# 
# Un **modelo de probabilidad** es una descripción de un evento aleatorio que consiste de dos partes:
# * La lista de todas las posibles salidas (espacio muestral).
# * Un probabilidad para cada salida.
# 
# La probabilidad de cualquier resultado es la frecuencia relativa a largo plazo de ese resultado.
# 
# ```{admonition} Calculo de la Probabilidad 
# Cuando las salidas del espacio muestral $S$ de un experimento son **igualmente probables**, la probabilidad de un evento $E$, denotada por $P(E)$, es la razon entre el numero de resultados asociadas a $E$  y el numero total de resultados del espacio muestral:
# 
# $$
# P(E) = \frac{Numero\; de\; salidas\; de\; E}{Numero\; de\; salidas\; de\; S}
# $$
# 
# ```
# Es importante anotar que hay dos tipos de resultados:
# * **Resultados igualmente probables**: Significa que cada resultado de un experimento ocurre con igual probabilidad.
# * **Eventos sesgados**: situación en la que los resultados no son igualmente probables.
# 
# Por otro lado, en lo que respecta a la probabilidad, esta cumple las siguientes propiedades:
# 
# ```{admonition} Algunas propiedades de la probabilidad
# 1. Para cualquier evento $E$, $0 \leq P(E) \leq 1$
# 2. Si $S$ es el espacio muestral de un experimento, $P(S) = 1$
# 3. Si un evento $E$ es imposible, entonces $P(E) = 0 $
# 4. Si $E$ es un evento seguro, entonces $P(E) = 1 $
# ```
# 

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


# ## Referencias
# 1. Statistics Openstax (https://openstax.org/details/books/statistics)
# 2. Introduction to Statistics and Data Analisys (Roxy Peck, Chris Olsen, Jay L. Devore)
# 3. The practice of Statistics (Starnes, Yates, Moore)
# 4. Probability and Stochastic Processes. A friendly introduction for Electrical and Computer Engineers (Yates, Goodman).
# 5. https://www.randomservices.org/random/index.html
# 6. https://bolt.mph.ufl.edu/6050-6052/
# 7. https://discovery.cs.illinois.edu/learn/Simulation-and-Distributions/Law-of-Large-Numbers/
# 8. https://discovery.cs.illinois.edu/learn/
# 9. https://www.randomservices.org/random/index.html
# 10. https://www.geogebra.org/m/UsoH4eNl
# 11. https://www.stapplet.com/
# 
# 
