---
layout: post
mathjax: true
title: Preprocesamiento-Entendimiento (1/2)
date: 2018-10-21 12:59 +0000
tags: [data science]
comments: true
---
 
Este es un post largo, con  un enfoque pensando para los que recién se inician
en el área de Data Science. Contiene descripciones detalladas de cada paso
realizado y ha sido dividido en 2. Siendo esta la primera parte.

## Introducción

Todas los proyectos que he realizado tienen, sin excepción, 2 cosas en común. El
preprocesamiento y entendimiento de datos. Para poder entender la importancia de
estas etapas, primero es necesario mostrarles que entiendo por cada una.

* Preprocesamiento: Son todos los procesos de transformación de la data, desde
su estado crudo (tal como se recibió), hasta que sea posible
analizarla/modelarla, con los modelos que se han propuesta para resolver el
problema.
* Entendimiento: Es la etapa de depuración y exploración de la data:
    * Para la fase de depuración buscamos resolver todos los datos
omitidos/erróneos, además de outliers que puedan estar presentes en los datos.
Es vital este paso, si esto no se realiza a conciencia, es posible que lleguemos
a conclusiones erróneas de nuestra.
    * En la fase de exploración se proponen y evalúan hipótesis que tenemos en
torno al problema en base a los datos. Aquí, se ocupan intensivamente distintas
técnicas de visualización. El rol que ellas cumplen es de lograr un panorama
general de la data. En nuestro caso, tratamos siempre evaluar la mayor cantidad
de hipótesis razonables posibles. Es muy frecuente vernos sorprendidos por
hipótesis que pensábamos se cumplían, en la realidad no lo hacían y viceversa.
Mi "futuro yo", siempre me lo ha agradecido.

**Nota 1** : El código que verán a continuación es
[Python](https://www.python.org/), quizás el lenguaje de programación más
utilizado en [Data Science](https://es.wikipedia.org/wiki/Ciencia_de_datos),
análisis de datos, etc. Próximamente, te mostraré como instalar y utilizar esta
herramienta en tu propio computador y algunas razones de porqué es tan
utilizada.

**Nota 2** : Los datos fueron obtenidos del excelente repositorio de la
[Universidad de California -
Irvine](https://archive.ics.uci.edu/ml/datasets/automobile). Se trata de un
caracterización de un pequeño listado de modelos de autos, junto con el precio
de venta. Este [dataset](https://es.wikipedia.org/wiki/Conjunto_de_datos) es
bastante sencillo y se eligió para ilustrar con mayor claridad las dos etapas en
las que se enfoca este post. 
 
## Plan:

* Carga Librerías
* Cargar Datos
* Limpieza
* Visualización 


{% highlight python %}
import numpy as np # para operaciones numéricas/matrices
import pandas as pd # para tablas de datos

#import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
import os #para funcionalidades relacionadas al sistema operativo (carpetas)
{% endhighlight %}
 
## Preprocesamiento 
 
### Carga 


{% highlight python %}
ruta_datos = os.getcwd() + "\\datos\\"

cols = ['symboling', 'norm_losses', 'make', 'fuel_type', 'aspiration', 
        'num_of_doors', 'body_style', 'drive_wheels',
       'engine_location','wheel_base', 'length', 'width', 'height', 
        'curb-weight', 'engine_type', 'num_of_cylinders', 'engine_size',
       'fuel_system', 'bore', 'stroke', 'compression_ratio', 'horsepower', 
        'peak_rpm', 'city_mpg', 'highway_mpg', 'price']

autos_df_orig = pd.read_csv(ruta_datos + 'auto.txt', names = cols)
len(autos_df_orig)
{% endhighlight %}




    205


 
Revisemos el código anterior:
La primera linea simplemente, recupera la ruta de la carpeta donde actualmente
está el script. A esta ruta le agregamos el string `\\datos\\` para obtener la
carpeta donde están ubicados los datos.
La segunda columna es necesaria por que Las columnas no vienen incluidas en el
dataset por lo que hay que agregarlas de forma separada.
La tercera, decimos que queremos leer los datos y almacenarlos en memoria (en
este caso, en un variable de tipo [dataframe](https://pandas.pydata.org/pandas-
docs/stable/generated/pandas.DataFrame.html) a la cual denominaremos
`auto_df_orig`).
Las última linea nos indica cuantas filas tiene el dataframe. 


{% highlight python %}
autos_df_orig.head(5).iloc[:,:5] 
{% endhighlight %}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>symboling</th>
      <th>norm_losses</th>
      <th>make</th>
      <th>fuel_type</th>
      <th>aspiration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
    </tr>
  </tbody>
</table>
</div>


 
Una vista rápida de las 5 primeras filas y columnas del dataframe. 
 
### Limpieza 


{% highlight python %}
cols_rl = [ 'price', 'horsepower', 'num_of_doors', 'body_style', 'length', 'width', 
           'num_of_cylinders', 'engine_size', 'peak_rpm']
{% endhighlight %}


{% highlight python %}
autos_df_orig[['horsepower', 'num_of_doors', 'price', 'peak_rpm']].head(5)
{% endhighlight %}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>horsepower</th>
      <th>num_of_doors</th>
      <th>price</th>
      <th>peak_rpm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>111</td>
      <td>two</td>
      <td>13495</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>111</td>
      <td>two</td>
      <td>16500</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>154</td>
      <td>two</td>
      <td>16500</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>102</td>
      <td>four</td>
      <td>13950</td>
      <td>5500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>115</td>
      <td>four</td>
      <td>17450</td>
      <td>5500</td>
    </tr>
  </tbody>
</table>
</div>


 
En la primera linea creamos un lista de las variables qué, a nuestro juicio,
podrían ser más interesantes de analizar.
En tanto que en la segunda mostramos las 5 primeras filas de las columnas
especificadas.

El siguiente comando nos permite, con la ayuda de la función anónima lambda y
concat, obtener un resumen de columnas que tienen (o no) el carácter `?` y a que
tipo de datos corresponde cada una (en pandas se le denomina `object` a las
columnas de tipo texto). 


{% highlight python %}
pd.concat([autos_df_orig[cols_rl].apply(lambda x: x.str.contains('\?').any()),
           autos_df_orig[cols_rl].dtypes],
          axis = 1, keys = ['contiene_?', 'tipo_dato'])
{% endhighlight %}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>contiene_?</th>
      <th>tipo_dato</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>price</th>
      <td>True</td>
      <td>object</td>
    </tr>
    <tr>
      <th>horsepower</th>
      <td>True</td>
      <td>object</td>
    </tr>
    <tr>
      <th>num_of_doors</th>
      <td>True</td>
      <td>object</td>
    </tr>
    <tr>
      <th>body_style</th>
      <td>False</td>
      <td>object</td>
    </tr>
    <tr>
      <th>length</th>
      <td>False</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>width</th>
      <td>False</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>num_of_cylinders</th>
      <td>False</td>
      <td>object</td>
    </tr>
    <tr>
      <th>engine_size</th>
      <td>False</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>peak_rpm</th>
      <td>True</td>
      <td>object</td>
    </tr>
  </tbody>
</table>
</div>


 
A partir de lo anterior, vemos que las columnas `horsepower`, `num_of_doors`,
`price` y `peak_rpm`, contienen este carácter. Dado ello, Python no le queda más
que establecer que todas aquellas columnas sean de tipo texto. Esto es
problemático, dado que por lo general, las columnas de tipo texto son más
limitadas y difíciles de analizar.

Por lo que es conveniente convertir estas columnas, que en realidad tienen
naturaleza numérica, a este tipo datos. Pero si es así, ¿Cómo es que debemos
convertir tales caracteres?

Ante el caso de datos erróneos u omitidos (como es en este ejemplo) podemos
hacer dos cosas, eliminarlas u imputarlas. La decisión de cual elegir y de qué
manera hacerlo, no es sencilla^ y esta fuera del alcance de este post. Por lo
que para esta ocasión, optaremos por la estrategia más sencilla (pero
probablemente errada) de solamente considerar las filas con datos válidos de
esas 4 columnas, almacenándola en un nuevo dataframe `auto_df`. Esto lo
realizamos con el siguiente comando: 


{% highlight python %}
autos_df = autos_df_orig[(autos_df_orig.horsepower != '?') & 
                         (autos_df_orig.num_of_doors != '?') & 
                         (autos_df_orig.price != '?') & 
                         (autos_df_orig.peak_rpm != '?')].copy()
{% endhighlight %}
 

^ Para mayor detención sugiero el libro [Machine Learning de K.
Murhpy](http://a.co/69aSfsp) (más técnico) o [Data Mining de Ian
Witten](http://a.co/2NdikLf). 
 
### Transformación 
 
El último paso restante es prestar atención a las columnas que están codificadas
como texto, pero debiesen estar como número:  `num_of_doors` y
`num_of_cylinders`.

Para las dos primeras se sigue una metodología similar. En el lado izquierdo
estamos creando una nueva columna `num_puertas` y `num_cilindros`, las cuales
corresponderán respectivamente a las columnas `num_of_doors` y
`num_of_cylinders`, pero donde estamos reemplazando cada cadena de texto por el
número correspondiente. Para esto último utilizamos la función `replace`, donde
le entregamos un diccionario donde las llaves del diccionario son el dato a
reemplazar, y los valores, con qué lo vamos a reemplazar.

La tercera línea es simplemente un listado de todas las columnas de naturaleza
numérica presentes en la data. La cual alimenta a la función del último código,
donde aplicamos la función de conversión `to_numeric` para quede del tipo de
datos adecuado. Acá utilizamos el parámetro `ignore`, que nos devuelve el mismo
input si este no se puede convertir. 


{% highlight python %}
autos_df['num_puertas'] = autos_df['num_of_doors'].replace(
    {"two": 2, "four": 4})

autos_df['num_cilindros'] = autos_df['num_of_cylinders'].replace(
    {"two": 2, "three": 3 , "four": 4, "five": 5,'six': 6, 'eight': 8, 
     'twelve': 12})

cols_num = [ 'price', 'horsepower', 'num_puertas', 'length', 'width', 
            'num_cilindros', 'engine_size', 'peak_rpm']

autos_df[cols_num] = autos_df[cols_num].apply(pd.to_numeric, errors='ignore')
{% endhighlight %}
 
Revisando el siguiente comando, vemos que para la columna `body_style` (o estilo
de chasis), necesitamos un tratamiento especial. Al ser una variable categórica,
necesitamos codificarla de alguna forma, para que las librerías de modelamiento
lo puedan entender (a ti te hablamos [sklearn](http://scikit-
learn.org/stable/)!). Aquí, utilizaremos la metodología más común y simple, [one
hot encoding](http://scikit-
learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).
Esto es que para cada una de esas categorías, crearemos una columna y en donde
codificaremos con un 1 cada vez que en la columna `body_style` aparezca esa
categoría, y un 0 en caso contrario. Esto `pandas` lo realiza de forma simple
con el comando `get_dummies` como se muestra en la última linea, devolviéndonos
un dataframe, la cual la guardamos en la variable `chasis_codif`. 


{% highlight python %}
autos_df['body_style'].value_counts()
{% endhighlight %}




    sedan          92
    hatchback      67
    wagon          24
    hardtop         8
    convertible     6
    Name: body_style, dtype: int64




{% highlight python %}
chasis_codif = pd.get_dummies(autos_df['body_style'])
{% endhighlight %}


{% highlight python %}
autos_df = pd.concat([autos_df, chasis_codif], axis = 1)
%store autos_df
%store cols_num
{% endhighlight %}

    Stored 'autos_df' (DataFrame)
    Stored 'cols_num' (list)
    
 
Finalmente, para unir ambos dataframe utilizamos la función `concat`. Esta
función toma un listado de series o dataframes y la concatena hacia al lado o
hacia abajo, coincidiendo ya sea las etiquetas de filas o columnas
respectivamente. Como en este caso, queremos concatenarlas hacia el lado
utilizamos `axis = 1`, si hubiese sido en caso contrario sería  `axis = 0`. Para
almacenar este dataframe concatenado, por conveniencia simplemente reutilizamos
la variable  `autos_df`. Luego  le indicamos a jupyter explícitamente que
queremos guardar el deataframe `autos_df'` y la lista `cols_num`, para así
utilizarla en el siguiente notebook. 
 
**Esto culmina la primera parte de este Post. En la segunda parte entraremos de
lleno a Entender la data.** 
