#!/usr/bin/env python
# coding: utf-8

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
# ### Media ponderada
# 
# En muchos casos no todos los valores tiene la misma importancia y puede ser útil otorgar pesos o valores a los datos dependiendo de su relevancia para determinado estudio. En ese caso no se suman los valores uno a uno sino se halla una medida conocida como **media ponderada**. 
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
