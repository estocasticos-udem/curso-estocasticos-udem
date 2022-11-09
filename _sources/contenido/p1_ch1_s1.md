# Introducción al Análisis Exploratorio de Datos (EDA)

## Introducción

El analisis estadistico de experimentos empieza con un analisis exploratorio de datos (EDA). El **EDA** (Exploratory Data Analysis) consiste en un conjuto de tecnicas de visualización y sintesis de datos para tratar de responder a una pregunta clave: **¿Que pueden decirnos los datos?**

El EDA es bastante util para cosas como:
* Detectar errores.
* Verificar supocisiones.
* Determinar relaciones entre variables.
* Resumir y presentar informacion.

El EDA fue planteado por Jonh W. Tukey introduciendo tecnicas estadisticas (mean, median, quantiles, etc.) y diagramas sencillos (boxplots, scatterplots, etc).

## Tipos de datos

Los datos proceden de muchas fuentes (medicion de sensores, eventos, texto, imagenes, videos, informacion de bases de datos, etc). La mayoria de estos datos cuando se toman de la fuente (**raw data**), no se encuentran procesados por lo que el primer objetivo consistira en procesar dichos datos para presentarlos de una manera apropiada (**structured data**) para su posterior analisis. Una de la formas mas usuales en las que se presentan los tipos de datos es mediante tablas con filas y columnas.

Inicialmente, es necesario conocer el tipo de datos a tratar para lo cual existen dos grandes grupos:
* **Numericos**: Datos que se expresan en una escala numerica. La siguiente tabla muestra los diferentes tipos de datos numericos:
  
  |Tipo|Definición|Sinonimo|Ejemplos|
  |---|---|---|---|
  |**Discretos**|Datos que se expresan en una escala numerica|entero, contable|Recuento de veces que ocurre un evento, edad, numero de clicks, numero de estudiantes, etc|
  |**Continuos**|Datos que tomar cualquier valor dentro de un intervalo|intervalo, real|Temperatura, estatura, velocidad del viento, indice de masa corporal, etc|

* **Categóricos**: Datos que solo pueden adoptar un numero especifico de valores que representan un conjunto de categorias posibles. A continuación se muestran diferentes tipos de datos categoricos:

  |Tipo|Definición|Sinonimo|Ejemplos|
  |---|---|---|---|
  |**Nominales**|Datos cuyos valores se asocian a categorias. Es importante decir que dentro de este tipo existe un caso especial que son los datos **binarios** los cuales se caracterizan por que solo pueden tomar dos posibles valores|entero, contable|Departamentos (Antioquia, Choco, Amazonas, etc.), ciudad (Medellin, Bogota,etc.), Tipos de pantallas TV (plasma, LCD, led, etc), sexo, si/no, Falso/Verdadero, Encendido/Apagado|
  |**Ordinales**|Datos categoricos que tienen un orden explícito|factor ordenado|Nivel académico, natisfacción de un servicio, estracto de los servicios publicos, etc|

```{Tip}
Conocer el tipo de datos es importante por que es la manera como se le indica al software como debe procesarlos.
```

## Datos estructurados

### Datos rectangulares

Los tipos de **datos rectangulares** (**rectangular data**) es la representación (estructura basica) de los datos para los modelos estadisticos. El termino general asociado a los datos rectangulares es una matriz bidimensional cuyas filas se asocian a los registros y sus columnas se asocian a las caracteristicas de cada registro. En aplicaciones como **R** y **Python**, el **data frame** (**Marco de datos**), es el formato específico para los **rectangular data**. 

La siguiente tabla resume los terminos claves asociados los datos rectangulares:

|Termino|Definición|Sinonimo|
|---|---|---|
|**Data Frame (Marco de dato)** |Son la estructura basica de datos (tablas) para los modelos estadisticos y de aprendizaje automatico||
|**Feature (Caracteristica)**|Columna de una tabla|attribute, input, predictor, variable|
|**Outcome (Resultado)**|Son el resultado de realizar el pronostico sobre los datos. A veces, las catacteristicas (features) se utilizan para pronosticar el resultado de un estudio.|dependent variable, response, target, output|
|**Records**|Fila dentro de una tabla|case, example, instance, observation, pattern, sample|

### Datos no rectangulares

Existen otro tipo de estructuras de datos mas complejas y variadas que las estructuras de datos rectangulares. Un ejemplo de estas estructuras son las series de tiempo y los datos espaciales por citar algunos.

La representación de datos asociados a este tipo e el **object**.

## Librerias Python para ciencia de datos

A continuación se muestran algunas librerias comunmente usadas en python para ciencia de datos.

### Toolboxes y librerias

|Libreria|URL|
|---|---|
|NumPy|[http://www.numpy.org/](http://www.numpy.org/)|
|SciPy|[https://www.scipy.org/scipylib](https://www.scipy.org/scipylib)|
|Pandas|[http://pandas.pydata.org/](http://pandas.pydata.org/)|
|SciKit-Learn|[http://scikit-learn.org/](http://scikit-learn.org/)|


### Librerias para visualización

|Libreria|URL|
|---|---|
|Matplotlib|[https://matplotlib.org/](https://matplotlib.org/)|
|Seaborn|[https://seaborn.pydata.org/](https://seaborn.pydata.org/)|

<!---

## Medidas

1. Metricas ([link](./metricas/metricas.ipynb))
2. Diagramas ([link](./diagramas/diagramas.ipynb))
-->

<!---
Whether you write your book's content in Jupyter Notebooks (`.ipynb`) or
in regular markdown files (`.md`), you'll write in the same flavor of markdown
called **MyST Markdown**.

## What is MyST?

MyST stands for "Markedly Structured Text". It
is a slight variation on a flavor of markdown called "CommonMark" markdown,
with small syntax extensions to allow you to write **roles** and **directives**
in the Sphinx ecosystem.

## What are roles and directives?

Roles and directives are two of the most powerful tools in Jupyter Book. They
are kind of like functions, but written in a markup language. They both
serve a similar purpose, but **roles are written in one line**, whereas
**directives span many lines**. They both accept different kinds of inputs,
and what they do with those inputs depends on the specific role or directive
that is being called.

### Using a directive

At its simplest, you can insert a directive into your book's content like so:

````
```{mydirectivename}
My directive content
```
````

This will only work if a directive with name `mydirectivename` already exists
(which it doesn't). There are many pre-defined directives associated with
Jupyter Book. For example, to insert a note box into your content, you can
use the following directive:

````
```{note}
Here is a note
```
````

This results in:

```{note}
Here is a note
```

In your built book.

For more information on writing directives, see the
[MyST documentation](https://myst-parser.readthedocs.io/).


### Using a role

Roles are very similar to directives, but they are less-complex and written
entirely on one line. You can insert a role into your book's content with
this pattern:

```
Some content {rolename}`and here is my role's content!`
```

Again, roles will only work if `rolename` is a valid role's name. For example,
the `doc` role can be used to refer to another page in your book. You can
refer directly to another page by its relative path. For example, the
role syntax `` {doc}`intro` `` will result in: {doc}`intro`.

For more information on writing roles, see the
[MyST documentation](https://myst-parser.readthedocs.io/).


### Adding a citation

You can also cite references that are stored in a `bibtex` file. For example,
the following syntax: `` {cite}`holdgraf_evidence_2014` `` will render like
this: {cite}`holdgraf_evidence_2014`.

Moreoever, you can insert a bibliography into your page with this syntax:
The `{bibliography}` directive must be used for all the `{cite}` roles to
render properly.
For example, if the references for your book are stored in `references.bib`,
then the bibliography is inserted with:

````
```{bibliography}
```
````

Resulting in a rendered bibliography that looks like:

```{bibliography}
```


### Executing code in your markdown files

If you'd like to include computational content inside these markdown files,
you can use MyST Markdown to define cells that will be executed when your
book is built. Jupyter Book uses *jupytext* to do this.

First, add Jupytext metadata to the file. For example, to add Jupytext metadata
to this markdown page, run this command:

```
jupyter-book myst init markdown.md
```

Once a markdown file has Jupytext metadata in it, you can add the following
directive to run the code at build time:

````
```{code-cell}
print("Here is some code to execute")
```
````

When your book is built, the contents of any `{code-cell}` blocks will be
executed with your default Jupyter kernel, and their outputs will be displayed
in-line with the rest of your content.

For more information about executing computational content with Jupyter Book,
see [The MyST-NB documentation](https://myst-nb.readthedocs.io/).

-->
