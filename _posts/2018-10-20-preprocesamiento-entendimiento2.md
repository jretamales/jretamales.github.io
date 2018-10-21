---
layout: post
mathjax: true
title: Preprocesamiento-entendimiento2
date: 2018-10-20 22:55 +0000
categories: data
comments: true
---
 
***Spoiler alert 1***: Esta es la segunda parte del post de
preprocesamiento/entendimiento. Para contextualizarlo, recomendamos leer el
[post anterior](http://teralab.cl/2018/04/19/2-etapas-que-no-pueden-faltar-en-
cualquier-proyecto-de-data-science/). En este post nos enfocaremos en
entendimiento. El código y los conceptos mostrados, están pensados para una
audiencia de nivel introductorio.

***Spoiler alert 2***: Por conveniencia y claridad, mostramos solo el código
referido a la etapa de entendimiento. Si se desean reproducir los resultados
mostrados en sus propios computadores, deberán necesariamente incluir el código
del [post anterior](http://teralab.cl/2018/04/19/2-etapas-que-no-pueden-faltar-
en-cualquier-proyecto-de-data-science/). 
 
## Plan:

* Carga Librerías
* Cargar Datos
* Limpieza
* Visualización 


{% highlight python %}
import numpy as np # para operaciones númericas/matrices
import pandas as pd # para tablas de datos

#import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
pd.options.display.float_format = '{:,.2f}'.format
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




{% highlight python %}
autos_df_orig.head(5)


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
      <th>num_of_doors</th>
      <th>body_style</th>
      <th>drive_wheels</th>
      <th>engine_location</th>
      <th>wheel_base</th>
      <th>...</th>
      <th>engine_size</th>
      <th>fuel_system</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression_ratio</th>
      <th>horsepower</th>
      <th>peak_rpm</th>
      <th>city_mpg</th>
      <th>highway_mpg</th>
      <th>price</th>
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
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.60</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.00</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>13495</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.60</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.00</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>hatchback</td>
      <td>rwd</td>
      <td>front</td>
      <td>94.50</td>
      <td>...</td>
      <td>152</td>
      <td>mpfi</td>
      <td>2.68</td>
      <td>3.47</td>
      <td>9.00</td>
      <td>154</td>
      <td>5000</td>
      <td>19</td>
      <td>26</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.80</td>
      <td>...</td>
      <td>109</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>10.00</td>
      <td>102</td>
      <td>5500</td>
      <td>24</td>
      <td>30</td>
      <td>13950</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.40</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.00</td>
      <td>115</td>
      <td>5500</td>
      <td>18</td>
      <td>22</td>
      <td>17450</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>


 
Revisemos el código anterior:
La primera linea simplemente, recupera la ruta de la carpeta donde actualmente
está el script. A esta ruta le agregamos el string `\\datos\\` para obtener la
carpeta donde están ubicados los datos.
La segunda columna es necesaria por que Las columnas no vienen incluidas en el
dataset por lo que hay que agregarlas de forma separada.
La tercera linea es simplemente para leer los datos y almacenarlos en memoria
(el dataframe `auto_df_orig` en este caso).
Las 2 últimas indica cuantas lineas están presentes en el dataframe y una vista
de sus 5 primeras líneas. 
 
### Limpieza 


{% highlight python %}
cols_rl = [ 'price', 'horsepower', 'num_of_doors', 'body_style', 'length', 'width', 
           'num_of_cylinders', 'engine_size', 'peak_rpm']

prueba = autos_df_orig[cols_rl].apply(lambda x: x.str.contains('\?').any())

prueba
{% endhighlight %}




    price                True
    horsepower           True
    num_of_doors         True
    body_style          False
    length              False
    width               False
    num_of_cylinders    False
    engine_size         False
    peak_rpm             True
    dtype: bool


 
En la primera linea creamos un lista de las variables que, a nuestro juicio,
podrían ser más interesantes de analizar.
Luego obtenemos una serie, con la ayuda de la función anónima lambda, de las
columnas que contienen el caractér `?`. Indicando, para cada columna, si en
algunas de sus filas contiene `?`.
Finalmente, mostramos esa serie en pantalla. 


{% highlight python %}
autos_df = autos_df_orig[(autos_df_orig.horsepower != '?') & 
                         (autos_df_orig.num_of_doors != '?') & 
                         (autos_df_orig.price != '?') & 
                         (autos_df_orig.peak_rpm != '?')].copy()
{% endhighlight %}
 
A partir de lo anterior, vemos que las columnas horsepower, num_of doors, precio
y peak_rpm, contienen el carácter `?`. La presencia de este carácter es
problemático, dado a que, por ejemplo con la columna precio, Python, como
encontró un carácter que no es un número, no le queda más que establecer que
toda la columna sean strings (cadenas de texto), limitando así, nuestra
capacidad para analizarla de manera más cuantitativa.

Ahora, ante datos erróneos u omitidos (como en este caso) podemos hacer dos
cosas, eliminarlas u imputarlas. La decisión de cual elegir y de qué manera
hacerlo, no es sencillo^ y esta fuera del alcance de este post. Por lo que para
esta ocasión, optaremos por la estrategia más sencilla (pero probablemente
errada) de solamente considerar las filas con datos válidos de esas 4 columnas,
almacenándola en un nuevo dataframe `auto_df`.

^ Para mayor detención sugiero el libro [Machine Learning de K.
Murhpy](http://a.co/69aSfsp) (más técnico) o [Data Mining de Ian
Witten](http://a.co/2NdikLf). 
 
### Transformación 
 
Como último paso procederemos a transformar los datos. Notar que solo
transformaremos los datos, de variables que tienen naturaleza numérica como
número de puertas y número de cilindros, pero por alguna razón están codificados
como texto. 


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
 
Para las dos primeras se sigue una metodología similar. En el lado izquierdo
estamos creando una nueva columna `num_puertas` y `num_cilindros`, las cuales
corresponderán respectivamente a las columnas `num_of_doors` y
`num_of_cylinders`, pero donde estamos reemplazando cada cadena de texto por el
número correspondiente. Para esto último utilizamos la función replace, donde le
entregamos un diccionario donde las llaves del diccionario son el dato a
reemplazar, y los valores, con qué lo vamos a reemplazar.

La tercera línea es simplemente un listado de todas las columnas de naturaleza
numérica presentes en la data. La cual alimenta a la función de la 4ta fila,
donde aplicamos la función de conversión to_numeric para quede el data type
adecuado. Acá utilizamos el parámetro `ignore`, que nos devuelve el mismo input
si este no se puede convertir. 


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
 
Revisando la primera linea, vemos que para la columna `body_style` (o estilo de
chasis), necesitamos un tratamiento especial. Al ser una variable categórica,
necesitamos codificarla de alguna forma, para que las librerías de modelamiento
lo puedan entender(a ti te hablo [sklearn](http://scikit-learn.org/stable/)).
Aquí, utilizaremos la metodología más común y simple, one hot encoding. Esto es
para cada una de esas categorias, crearemos una columna y en donde codificaremos
con un 1 cada vez que en la columna `body_style` aparezca esa categoría, y un 0
en caso contrario. Esto `pandas` lo realiza de forma simple con el comando
`get_dummies` como se muestra en la última linea, devolviéndonos un dataframe,
la cual la guardamos en la variable `chasis_codif`. 


{% highlight python %}
autos_df = pd.concat([autos_df, chasis_codif], axis = 1)
{% endhighlight %}
 
Finalmente, para unir ambos data frame utilizamos la función `concat`. Esta
función toma un listado de series o dataframes y la concatena hacia al lado o
hacia abajo, coincidiendo ya sea las etiquetas de filas o columnas
respectivamente. Como en este caso, queremos concatenarlas hacia el lado
utilizamos `axis = 1`, si hubiese sido en caso contrario sería  `axis = 0`. Para
almacenar este dataframe concatenado, por conveniencia simplemente reutilizamos
la variable  `autos_df`. 
 
## Entendimiento de la data 
 
Antes de seguir, detengámonos por un momento para plantear algunas hipótesis en
torno al problema. Nuestro objetivo, es poder llegar a obtener un entendimiento
general de la data, a través de la evaluación de estas hipótesis en virtud de
ella. Todas las hipótesis girarán al problema principal que hemos planteado para
este dataset, cuales atributos son los contribuyen a un mayor precio de un
vehículo? *(Nota: Este es solo uno de los problemas posibles. Para un mismo
dataset siempre podemos plantear más de un problema a abordar. Esto también, es
parte del encanto de esta disciplina. Siempre tendremos desafíos por resolver)*:


* Recordemos un poco cuales son las variables en que hemos querido enfocarnos:
(`horsepower`, `num_puertas`, `width`, `length`, `num_cilindros`, `engine_size`,
`peak_rpm`, `body_style`) y analicemos cada una de forma separada:
    * `horsepower`: Es el poder máximo del motor. Mientras mayor esta métrica,
más capacidad de aceleración tendrá el vehículo. Así, debemos suponer que este
es un elemento deseable para los consumidores, por lo que esperamos un mayor
precio a medida que este aumente.
    * `num_puertas`: Acá es un poco más difuso. Los convertibles por lo general
tienen 2 puertas, por lo que si suponemos que todos los vehículos de dos puertas
son convertibles, veríamos una relación negativa de numero de puertas vs precio.
Sin embargo, este no es el caso. Por ejemplo, también existen city cars que
también tienen 2 puertas, que son mucho más económicos. Probablemente así,
esperaremos no ver una relación muy fuerte entre `num_puertas` y `price`.
    * `width`, `length`: Los camiones y SUV, son autos de mayor tamaño y por lo
general, más costosos. Sin embargo, no siempre se da que los vehículos de mayor
tamaño, sean más costosos. Un contraejemplo, también resultan ser los
convertibles.
    * `num_cilindros`: La [cilindrada](https://es.wikipedia.org/wiki/Cilindrada)
$(C)$ viene a ser una suerte de capacidad pulmonar de los vehículos. Dónde a
mayor cilindrada, mayor capacidad del vehículo, (y probablemente mayor precio).
La cilindrada se calcula con $C = V*N$. Con $V$, volumen del cilindro y $N$
número de cilindros. Viendo tal ecuación, vemos que no siempre los autos que
tengan mayor $N$ o `num_cilindros`, tengan mayor cilindrada. Porque puede
suceder, que los cilindros de cada vehículo, tengan distinto volumen.
    * `engine_size`: Esto es justamente la cilindrada. Por lo que esperamos un
relación más estrecha (y positiva) entre esta métrica y el precio vs
`num_cilindros`.
    * `peak_rpm`: Acá, de seguro demostramos que no somos expertos en el tema (y
corríjanos en los comentarios si nuestro entendimiento es errado). Pero a
grandes rasgos nos habla del nivel de rpm, donde el motor funciona a máximo
poder. Así, como rpm, está relacionada con la acelaración del vehículo, a mayor
`peak_rpm` tendremos un mayor capacidad de obtener un buen póder y aceleración
al mismo tiempo, lo cual es deseable para el comprador. Sin embargo, al [leer un
poco más](https://en.wikipedia.org/wiki/Power_band), vemos que no siempre es
así. Por ejemplo, puede suceder que ese peak solo sea alcanzado en un rango muy
estrecho de rpm, por lo que alcanzar este máximo de desempeño sea muy difícil de
alcanzar.
    * `body_style`: Se refiere al tipo de chasis. Esperamos que para esta
métrica, los automóbiles del tipo convertible (`convertible`, `hardtop`) tengan
mayor precio que los otros. 
 
### Métricas de tendencia central: 


{% highlight python %}
autos_df[cols_num].describe()
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
      <th>price</th>
      <th>horsepower</th>
      <th>num_puertas</th>
      <th>length</th>
      <th>width</th>
      <th>num_cilindros</th>
      <th>engine_size</th>
      <th>peak_rpm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>197.00</td>
      <td>197.00</td>
      <td>197.00</td>
      <td>197.00</td>
      <td>197.00</td>
      <td>197.00</td>
      <td>197.00</td>
      <td>197.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>13,279.64</td>
      <td>103.60</td>
      <td>3.14</td>
      <td>174.22</td>
      <td>65.89</td>
      <td>4.37</td>
      <td>126.99</td>
      <td>5,118.02</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8,010.33</td>
      <td>37.64</td>
      <td>0.99</td>
      <td>12.37</td>
      <td>2.12</td>
      <td>1.07</td>
      <td>41.91</td>
      <td>481.04</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5,118.00</td>
      <td>48.00</td>
      <td>2.00</td>
      <td>141.10</td>
      <td>60.30</td>
      <td>2.00</td>
      <td>61.00</td>
      <td>4,150.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7,775.00</td>
      <td>70.00</td>
      <td>2.00</td>
      <td>166.80</td>
      <td>64.10</td>
      <td>4.00</td>
      <td>97.00</td>
      <td>4,800.00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>10,345.00</td>
      <td>95.00</td>
      <td>4.00</td>
      <td>173.20</td>
      <td>65.50</td>
      <td>4.00</td>
      <td>119.00</td>
      <td>5,200.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>16,503.00</td>
      <td>116.00</td>
      <td>4.00</td>
      <td>183.50</td>
      <td>66.90</td>
      <td>4.00</td>
      <td>145.00</td>
      <td>5,500.00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>45,400.00</td>
      <td>262.00</td>
      <td>4.00</td>
      <td>208.10</td>
      <td>72.00</td>
      <td>12.00</td>
      <td>326.00</td>
      <td>6,600.00</td>
    </tr>
  </tbody>
</table>
</div>


 
Con la anterior línea podemos devolver un resumen general de las principales
métricas de cada variable.
Viendo `price` vemos que su media está más cargada hacia valores bajos, donde el
75% de las observaciones bajo $45,400. Algo parecido sucede con `horsepower`,
donde el 75% de las observaciones bajo 116. Las variables `length` y `width`, al
parecer se comportan de una forma más parecida a una normal. 
 
### Correlación de Pearson: 


{% highlight python %}
autos_df[cols_num].corr().loc['price', cols_num[1:]]
{% endhighlight %}




    horsepower       0.81
    num_puertas      0.05
    length           0.69
    width            0.75
    num_cilindros    0.71
    engine_size      0.87
    peak_rpm        -0.10
    Name: price, dtype: float64


 
La linea anterior nos muestra la correlación de Pearson de todas la variables
`cols_num` vs `price`. Para destacar, es que a través de esta métrica, medimos
el grado de relación lineal entre cada variable y precio.  Que una correlación
sea 0, no necesariamente significa que esta variable no tenga relación con el
precio. Solo que esta relación, si es que existe, es no lineal.
Aún así, vemos que las variables `horsepower`, `length`, `width`, `horsepower`,
`engine_size` y `num_cilindros`, presentan una alta correlación con `price`.
Tomando un paso más, veamos si visualmente estas relaciones son evidentes. 
 
### Visualización 


{% highlight python %}
from pandas.plotting import scatter_matrix
scatter_matrix(autos_df[cols_num], figsize=(10, 10));
{% endhighlight %}


![png](/assets/images_files/2018-10-20-preprocesamiento-entendimiento2_31_0.png)

 
La primera línea de la celda anterior, es una simple importación de la función
de matriz de dispersión. Los gráficos de dispersión, son ampliamente usados
cuando las variables en cuestión son cuantitativas, y permiten teorizar de buena
forma, como se relaciona una variable versus la otra.

Mientras que en la segunda, aplicamos tal función. El resultado de esta, es una
matriz donde todas los cuadros, salvos los de la diagonal son gráficos de
dispersión. La diagonal en cambio, para no mostrar un gráfico de dispersión
redundante se muestra un histograma, que viene a representar de forma aproximada
la distribución de cada variable.

Fijándonos solo en la primera fila, (la que tiene como eje `y` precio) vemos
cosas interesantes. A la vez que `engine_size`, `width` y `num_cilindros`,
muestran una relación marcadamente
[lineal](https://es.wikipedia.org/wiki/Lineal) el relación a  `price`, variables
como `horsepower` o `length`, muestran una tendencia a una relación más bien
[cuadrática](https://es.wikipedia.org/wiki/Funci%C3%B3n_cuadr%C3%A1tica). Este
hallazgo entonces, nos abre una puerta. Que tal si en vez de mirar una relación
del tipo $y = x$, transformamos x para que investigar $y = x^2$? Esto lo
dejaremos, quizás para un próximo post. 


{% highlight python %}
ax = autos_df[['price', 'body_style']].boxplot( by = 'body_style')
ax.set_title("")
ax.get_figure().suptitle("Price vs body_style")
_= ax.set_ylabel('Price')
{% endhighlight %}


![png](/assets/images_files/2018-10-20-preprocesamiento-entendimiento2_33_0.png)



{% highlight python %}
ax = autos_df[['price', 'num_puertas']].boxplot( by = 'num_puertas')
ax.set_title("")
ax.get_figure().suptitle("Price vs Num_puertas")
_ = ax.set_ylabel('Price')
{% endhighlight %}


![png](/assets/images_files/2018-10-20-preprocesamiento-entendimiento2_34_0.png)

 
Como análisis final, se utilizan un tipo de gráficos que tienen quizás, el mejor
nombre de todos: Caja y Bigotes. Este tipo de gráficos es especialmente útil
para una entender la relación de una variable cualitativa u ordinal, versus una
cuantitativa. De modo muy breve, cada caja (y bigote) muestra la distribución de
la variable `y` versus un valor o categoría de la variable `x`. La linea
inferior de la caja marca el 25% de los datos, la linea verde el 50% y la linea
superior de la caja, el 75%. Las lineas horizontales de los bigotes marcan el
mínimo y máximo, sin considerar los outliers que se representan por pequeños
círculos o puntos.

Considerando lo anterior, en el primer gráfico vemos una clara tendencia a que
los vehículos del tipo `convertible` o `hardtop`, tengan un mayor precio,
mientras que los otros 3 tipos de chasis presentan un menor precio, sin haber
tanta diferencia entre uno y otro salvo en el caso, del tipo sedan que muestra
una mayor variabilidad.

En cuanto al segundo gráfico, no se ve mayor efecto del precio en virtud del
número de puertos. 
 
## Comentarios Finales

Aquí termina nuestro primer viaje introductorio por el preprocesamiento y
entendimiento de datos. El resultado de esto es haber obtenido conocimiento
frente a un dataset, del cual solo teníamos hipótesis no validadas. Cómo habrán
claramente notado, el desarrollo de estas etapas no fueron del todo exhaustivas.
Hay pasos al cual no le asignamos el tiempo debido (por ejemplo: datos erróneos
y omitidos) y conceptos que se podrían haber ilustrado de mejor manera (one-hot
encoding). Esto fue absolutamente intencional, nuestro deseo era mostrar de
principio a fin, los pasos más importantes de este proceso de principio a fin.
Más adelante trataremos de detenernos con más detalle en cada paso individual. 
