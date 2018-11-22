---
layout: post
mathjax: true
title: Un truquito de Sklearn... Pipelines! (1/2)
date: 2018-11-08 12:20 +0000
tags: [sklearn, pipelines, machine learning]
comments: true
---
 
# Temas a tratar: 
 
Este es el primer post de la serie dedicada a un truco o metodología (si así lo
quisieran llamar) de Sklearn: [Pipelines](http://scikit-
learn.org/stable/modules/compose.html#). Una herramienta tremendamente útil de
la librería Sci-kit Learn ([`sklearn`](http://scikit-learn.org/)) que me
gustaría haber conocido hace mucho antes. Para sacar el máximo provecho a estos
posts se sugiere que el lector previamente:
* Haya alguna vez, ajustado un modelo con `sklearn` y que especialmente le sea
familiar los métodos `fit` y `transform` de esta librería.
* Haber utilizado `pandas` como herramienta de preprocesamiento y transformación
de data, de manera similar a lo relatado en mi post anterior de
[preprocesamiento y
entendimiento](https://jretamales.github.io/2018-10-21-preprocesamiento-
entendimiento1/).
* Entienda a que nos referimos con los conceptos de validación cruzada. Y,
preferiblemente, haya alguna alguna vez utilizado el método
[GridSearchCV](http://scikit-
learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) o
[RandomizedSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.mo
del_selection.RandomizedSearchCV.html)
* Entienda como crear Clases en Python. 
 
# Introducción 
 
El primer paso para extraer todo el potencial de Pipelines es entender las
*Transformaciones customizadas*, objetivo principal de este post. Sin embargo,
estas serán muy dificiles de entender, sin explicar el rol que cumplen las
pipelines de sklearn, en las etapas que prácticamente todo proyecto de machine
learning deben tener: Lectura, Procesamiento de variables, Ajuste del modelo y
Evaluación de resultados.

En concreto, los pipelines nos permiten encapsular cada una de estas etapas (con
excepción de la lectura de datos) en un solo gran proceso para así considerar el
pipeline resultante como un gran estimador de sklearn. Dejame explicarte a que
me refiero con esto.

Para ello me gustaría que imagines el siguiente escenario. Tienes a tu haber un
dataset con dos variables, ambas numéricas, una independiente, a la que
denominaremos `x` y una dependiente, `y`. Tu objetivo es encontrar un modelo,
donde solo mediante `x` podamos predecir o estimar con buena precisión el valor
que tomará `y`, en el caso, obviamente, que este último ya no lo tengamos. Ante
tal problema, la principal pregunta que nos gustaría resolver es ¿Qué
combinación de procesamiento y modelo estadístico, nos permitirá obtener el
mejor resultado posible?

Para resolver tal desafío, supongo que utilizarás una metodología muy similar a
lo que la gran mayoría de los que usuarios de Python harían. En específico
utilizarías `pandas` para manipular o transformar variables y `sklearn` para el
ajuste de modelos. Individualmente, las API de cada uno están pensados para que
cada tarea te demande el menor tiempo posible, reduciendo la cantidad de código
a escribir. Por lo que no sorprende, que ambas sean ampliamente las herramientas
más utilizadas para este desafío.

No obstante, me gustaría convencerte que nuestro problema no radica ahí. Al
contrario, radica principalmente en que tanto para la etapa de procesamiento y
modelos tenemos una serie de opciones posibles, los que de manera combinada, nos
entregan un gran abanico de opciones. Si esto no es suficiente para clarificar
mi idea, volvamos al ejemplo anterior: Imaginando que además de tomar `x` en la
misma forma como venía originalmente, nos gustaría probar con diferentes
transformaciones: Específicamente, con $x^2$, $x^{-1}$, $log(x)$ , $e^x$ y
$\sqrt{x}$, esto para la etapa de procesamiento de variables. A su vez, para el
Ajuste del modelo, creemos que tanto [Regresión lineal](http://scikit-
learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
como [Lasso](http://scikit-
learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html),
[Ridge](http://scikit-
learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) y [Elastic
Net](http://scikit-
learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html) parecen
a priori como buenas opciones. Si, consideramos todas las combinaciones de
procesamiento vs ajuste como validas, nos encontramos que debiésemos repetir 20
(5 transformaciones x 4 modelos) veces el proceso combinado de transformación en
`pandas` y luego estimación en `sklearn`. Como correctamente supondrás, si
utilizas independientemente ambas librerías como la gran mayoría de los usuarios
haría, deberás entonces construir un laaaargo script, donde tendrás que dedicar
especial esfuerzo en organizar y resumir el resultado de cada modelo en forma
ordenada, para así tener cierto control sobre cada experimento (combinación
preprocesamiento-ajuste). Este tipo de situaciones, es justamente lo que las
`sklearn pipelines` nos ayudan a solucionar.

En términos sencillos, por medio del uso de pipelines, buscamos modularizar las
etapas mencionadas anteriormente, con una una lógica estándar para luego
concatenarlas de manera secuencial. Esto, finalmente, nos permitirá hacer uso de
las funciones `fit`  y `predict` -- los verdaderos motores analíticos de
`sklearn` -- para el proceso completo y no solamente para el ajuste del modelo.
Como valor adicional, obtendremos un código más legible, donde cada paso del
proceso queda claramente explicitado y donde encontrar la combinación correcta
de estrategia de procesamiento + modelo, sera revelerá de forma más directa. 
 
# Custom Transformers 
 
Así, hemos llegado al tema central de este post: Transformaciones de Dataset
Customizadas, elemento clave para poder traducir las transformaciones realizadas
en `pandas` a una transformación que `sklearn` la pueda entender. Sin este
proceso de traducción, `sklearn` no podrá integrar el procesamiento customizado
para nuestro problema al Pipeline y obtener los beneficios anteriormente
mencionados. (*Nota: Cabe mencionar, que no necesariamente se debe definir  un
**nuevo** objeto de procesamiento. De hecho, `sklearn` dispone de múltiples
funciones de procesamiento, que pueden resolver una gran variedad de problemas.
En efecto, este post busca ilustrar como podemos extender aún más esas
capacidades, generando una propia para ajustarse a nuestras necesidades
particulares).*

De forma breve, las `Transformaciones Customizadas` son un objeto que nosotros
definimos. La cual, para que pueda conversar con sklearn, debe contar como
mínimo con 3 funciones `__init__`, `fit` y `transform`. Además de ello, por lo
general este objeto lo creamos a partir de un objeto de bajo nivel de
`sklearn`: `TransformerMixin`. Heredando desde este último, nos permitirá luego
hacer uso del método `fit_transform` de manera gratuita. Más adelante explicaré
por que esto es conveniente. Antes de ello, lo mejor es comprender cada función
mencionada elementalmente:
* `__init__`: Para crear un nuevo objeto de transformación con los atributos que
queremos que este posea.
* `fit`: Obtiene los parámetros a partir de la data (cómo mediana, desviación
estándar, media, etc), necesarios para poder transformar la data. En las
ocasiones en que no es necesario estimar un parámetro a partir de la data (cómo
en el caso de $x^2$, $x^{-1}$, $log(x)$ , $e^x$ y $\sqrt{x}$. Esta función
debiese devolver los mismos datos de entrada.
* `transform`: Es la función que transforma la data a partir de los parámetros
estimados en `fit` (o ninguno si es así el caso) y de la transformación
específica que nosotros deseemos definir. 
 
## Código 
 
A continuación, mostramos como creamos un objeto de transformación simple de
nuestro dataset. A este le llamaremos `ModeImputer`. El cuál reemplazará, para
cada columna, las celdas que correspondan a un carácter que nosotros
determinaremos por la  moda de cada columna. Cómo procederemos es más bien
simple, importamos las librerías, leemos la data, luego definimos `ModeImputer`,
para finalmente ejemplificar como podría utilizarse. 


{% highlight python %}
# Importando librerías
import pandas as pd
from sklearn.base import TransformerMixin # el objeto de sklearn en que 
#  nos basaremos para transformar los datos.

from sklearn.model_selection import train_test_split # metodo de sklearn
# para separar nuestro dataset en entrenamiento y testing

import numpy as np
{% endhighlight %}
 
Leemos los datos, especificando los nombres de columnas. Al igual que en
[preprocesamiento y
entendimiento](https://jretamales.github.io/2018-10-21-preprocesamiento-
entendimiento1/), utilizaremos el mismo dataset de `auto` disponible en [UCI
Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php) 


{% highlight python %}
p = './datos/auto.txt'
cols = ['symboling', 'norm_losses', 'make', 'fuel_type', 'aspiration', 
        'num_of_doors', 'body_style', 'drive_wheels',
       'engine_location','wheel_base', 'length', 'width', 'height', 
        'curb-weight', 'engine_type', 'num_of_cylinders', 'engine_size',
       'fuel_system', 'bore', 'stroke', 'compression_ratio', 'horsepower', 
        'peak_rpm', 'city_mpg', 'highway_mpg', 'price']
df = pd.read_csv(p, names= cols)

{% endhighlight %}
 
Finalmente definimos nuestro objeto de Transformación de Dataset Customizada. 


{% highlight python %}
class ModeImputer(TransformerMixin):
    
    def __init__(self, missing_value = '?'):
        self.missing_value = missing_value # Espeficamos el atributo con el que 
        # debe ser inicializado el objeto.
    
    def fit(self, X, y=None):
        # En la función fit es donde obtenemos los parametros que nos
        # para luego hacer la imputación. En este caso será un diccionario 
        # de reemplazo.
        
        # primero buscamos las columnas que contengan 
        #el cáracter especificado.
        cols = X.apply(lambda x: x.str.contains('\\' + self.missing_value).any())
        # luego reemplazamos las celdas con ese caracter con np.nan
        df_tmp = X.loc[:,cols].replace({self.missing_value: np.nan})
        
        # finalmente, para esas columnas obtenemos la moda para cada una
        # esto lo guardamos en un diccionario dentro del mismo objeto
        # ModeImputer
        self.replace_with = dict(zip(df_tmp.columns, df_tmp.mode(axis = 0).values[0]))
        return self
    
    def transform(self, X):
        # Transform lo único que hace es tomar el diccionario computado y 
        # devolver el dataframe de vars independientes reemplazado.
        replace_with = {k:{self.missing_value: v} for k, v in self.replace_with.items()}
        
        return X.replace(to_replace = replace_with)
{% endhighlight %}
 
Con lo anterior, hemos definido todo lo necesario para poder crear un objeto de
transformación de dataset compatible con sklearn. Como punto importante, es el
lugar en la clase en donde obtenemos los parámetros de la data -- el cual en
este caso es el diccionario de remplazo `self.replace_with`. Donde se lo
escribimos cómo parte de la función `fit` y no de `transform`.

Así, la transformación de los datos de tanto entrenamiento como testing para
estos parámetros, solo se realizará con los parámetros calculados a partir de
los datos de entrenamiento y no los de testing. Este razonamiento, se basa en la
distinción que debemos tener entre datos de entrenamiento vs testing. Donde los
primeros son utilizados para ajustar el modelo y el segundo para evaluarlo.
Buscando siempre que esta evaluación refleje lo que sucede en la práctica, que
no tenemos acceso a los datos de entrenamiento, más que para solo evaluar el
modelo.

En este sentido, un error común, es utilizar parte de datos de testing, para
calcular parámetros con que imputaremos o transformaremos el dataset de
entrenamiento,que es el cual donde nos basamos para ajustar el modelo. Este
error, da como resultado un modelo contaminado, dado que dentro de los datos
utilizados para ajustarlo se incluyeron datos de testing. 
 
A continuación un breve ejemplo de como se utilizaría `ModeImputer`. Primero
separamos nuestro dataset `df`, en entrenamiento y testing. Luego pasamos
ajustar e imputar (transformar) nuestra data de entrenamiento. Para finalmente
imputar nuestra dataset de testing con los parámetros estimados en la linea
anterior. 


{% highlight python %}
# train test split siempre entrega las separaciones en este orden.
X_train, X_test, y_train, y_test = train_test_split(df[df.columns[:-1]],
                                                    df['price'] )
# Nuestro objeto para imputar.
mr = ModeImputer(missing_value = '?')
{% endhighlight %}
 
Primero ajustamos a la data para obtener los parametros de reemplazo y luego
imputar el dataset de entrenamiento. Notar cómo `ModeImputer` hereda de
`TransformerMixin`, también podriamos haber hecho el ajuste y transformación en
una sola línea. Así: `X_train_transf = mr.fit_transform(X = X_train)` 


{% highlight python %}
mr.fit(X = X_train) #Ajuste para obtener los parametros de reemplazo
X_train_transf = mr.transform(X = X_train)# Tranformación
{% endhighlight %}
 
Podemos inspeccionar cuales son los valores de reemplazo para cada columna con: 


{% highlight python %}
mr.replace_with
{% endhighlight %}




    {'norm_losses': '161',
     'num_of_doors': 'four',
     'bore': '3.62',
     'stroke': '3.40'}


 
Finalmente transformamos nuestro data de testing. En este caso no es necesario
hacer `fit` nuevamente, ya que se realizó para el entrenamiento. 


{% highlight python %}
X_test_transf = mr.transform(X = X_test) 

{% endhighlight %}
 
Cómo ultimo código verifiquemos que no es lo mismo tranformar los datos de
testing con los parametros obtenidos a partir de los datos de entrenamiento
(como lo hicimos en el código anterior) vs hacerlo con los propios datos de
entrenamiento. Esto lo realizaremos utilizando la función de pandas `equals`,
que verifica si 2 df son iguales. 


{% highlight python %}
X_test_transf.equals(mr.fit_transform(X = X_test))
{% endhighlight %}




    False


 
El codigo anterior nos ilustra la claridad que podemos lograr con esta
metodología. Usando `mr.fit()` y `mr.transform()`, al igual que lo hacemos para
ajustar el modelo. Mirándolos por un par de segundos, no es difícil imaginarse
como podríamos customizar y reutilizarlos para otro tipo de problemas, siguiendo
la misma metodología usada comunmente por `sklearn`.

Si aún, para ti, esto no es tan claro, lo entiendo y creo pensar que no eres el
único -- de hecho ni siquiera importamos el módulo `Pipeline`, que tanto
destacábamos. Aún así, tengo la esperanza que en el siguiente post esto hará más
sentido. Allí, construiremos a partir de la bases que aquí hemos descrito,
describiendo más explícitamente como se combinan las etapas de procesamiento de
variables, ajuste y evaluación del modelo en un solo gran proceso. 
