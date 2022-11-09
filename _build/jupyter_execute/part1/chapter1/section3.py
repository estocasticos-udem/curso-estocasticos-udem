#!/usr/bin/env python
# coding: utf-8

# # Diagramas
# 
# En la estadistica es muy común el uso de representaciones visuales para mostrar el comportamiendo de los datos, las cuales seran mostradas a continuación.
# 
# ## Representación de datos numericos
# 
# 
# ### Histograma
# 
# Un histograma es una representación gráfica de una variable en forma de barras en la cual la superficie de la barra es proporsional a la frecuencia de los valores representados. Son utiles por que sirven para obtener un panorama general de la forma como se distribuyen los datos. La construcción de un histograms depende del tipo de datos que se esten empleando (discretos o continuos) tal y como de describe a continuación:
# * **Datos discretos**: En primer lugar, se determinan la frecuencia y la frecuencia relativa de cada valor $x_i$. Luego se marcan los valores $x_i$ posibles en una escala horizontal. Sobre cada valor se traza un rectángulo cuya altura es la frecuencia relativa (o alternativamente, la frecuencia) de dicho valor: Los rectángulos deben medir lo mismo de ancho.
# * **Datos Continuos**: En este caso, se divide el el eje de medición entre un número adecuado de *intervalos de clase o clases*. Cuando el ancho de las clases es el mismo, lo que se hace es determinar la frecuencia y la frecuencia relativa de cada clase contando la cantidad de muestras que se encuentran dentro de cada intervalo y usando frecuencia relativa como la altura del correspondiente intervalo.
# 
# 
# 
# ## Markdown + notebooks
# 
# As it is markdown, you can embed images, HTML, etc into your posts!
# 
# ![](https://myst-parser.readthedocs.io/en/latest/_static/logo.png)
# 
# You can also $add_{math}$ and
# 
# $$
# math^{blocks}
# $$
# 
# or
# 
# $$
# \begin{aligned}
# \mbox{mean} la_{tex} \\ \\
# math blocks
# \end{aligned}
# $$
# 
# But make sure you \$Escape \$your \$dollar signs \$you want to keep!
# 
# ## MyST markdown
# 
# MyST markdown works in Jupyter Notebooks as well. For more information about MyST markdown, check
# out [the MyST guide in Jupyter Book](https://jupyterbook.org/content/myst.html),
# or see [the MyST markdown documentation](https://myst-parser.readthedocs.io/en/latest/).
# 
# ## Code blocks and outputs
# 
# Jupyter Book will also embed your code blocks and output in your book.
# For example, here's some sample Matplotlib code:

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
