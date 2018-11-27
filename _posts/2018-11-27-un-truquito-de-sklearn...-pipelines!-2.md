---
layout: post
mathjax: true
title: Un truquito de Sklearn... Pipelines! (2/2)
date: 2018-11-27 11:30 +0000
tags: [sklearn, pipelines, machine learning]
comments: true
---
 
# Introducción 
 
Llegó la hora de intentar cumplir nuestra promesa del post anterior. Esto
significa que trataremos de ilustrar, mediante código, el valor que entrega
`Pipelines` en problemas de Machine Learning.

A manera de ser aún más convincente, extenderé levemente la clase de imputación
de datos `ModeImputer` que habíamos creado en el post anterior, a la vez de
conectarlo con otros dos procesos más: Transformación de variables y
Modelamiento supervisado. Mediante esto mostraré, lo que es para mí el principal
aspecto diferenciador de `Pipelines`: La capacidad de evaluar de forma simple  y
eficiente, todas las combinaciones posibles de las variaciones de cada método
que hemos definido en nuestro `Pipeline`. La forma en que haremos esto es
mediante los conocidos métodos de `sklearn` `GridSearchCV` o
`RandomizedSearchCV`, los cuales son ampliamente utilizados para buscar la
combinación óptima de parámetros de un modelo en virtud de los datos de
entrenamiento. El único ingrediente extra en nuestro caso, es que consideramos
la cadena de método de imputación, transformación y modelo supervisado, como un
solo gran estimador, utilizando `GridSearchCV` o  `RandomizedSearchCV`
directamente sobre el. 
 
----- 
 
# Código 
 
El procedimiento es como sigue: Primero importaremos todas las librerías que
utilizaremos en el transcurso del post.
Luego se presenta `ModeImputer` extendido, al cual le cambiamos a un nombre más
general: `MyImputer`. Como 3era etapa generaremos datos ficticios, ilustrando el
patrón que debiésemos esperar que el modelo encuentre. Finalmente, crearemos
nuestro `Pipeline` para luego definir que parámetros deseamos que `sklearn`
considere para calibrar y ajustaremos el `Pipeline` mostrando directamente la
combinación óptimo de parámetros-modelo. 
 
## Importar Librerías. 


{% highlight python %}
# Importamos librerías, recurriendo fuertemente a modulos de sklearn, 
# el enfoque principal de nuestro post además de matplotlib 
# para presentar de forma más clara el patrón de los datos considerados.
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
import matplotlib.pyplot as plt
%matplotlib inline
{% endhighlight %}
 
## Clase `ModeImputer `extendida 


{% highlight python %}
class MyImputer(BaseEstimator, TransformerMixin):
    """
    Nuestra Clase anterior modificada. El principal 
    cambio a destacar es que se extiende la clase 
    para entregarle al usuario la opción de elegir 
    la metodología con que se imputarán el dataframe 
    dentro de dos opciones: `mode` (moda) y `mean` (media). 
    Implicando que `ModeImputer` ya no es el nombre más adecuado.
    Además de lo anterior, tenemos que llamar dentro de la clase 
    al módulo `BaseEstimator` que entrega a MyImputer 
    funciones que `Pipelines` necesariamente requiere.
    """
    
    def __init__(self, missing_value = '?', impute_type = 'mode'):
        self.missing_value = missing_value
        self.impute_type = impute_type
    
    def fit(self, X, y=None):
        # identificamos las columnas con carácter problematico
        cols = X.apply(lambda x: x.str.contains('\\' + self.missing_value).any()).copy()
        df_tmp = X.loc[:,'X'].replace({self.missing_value: np.nan})
        # convertimos a numeros y luego a dataframe.
        df_tmp = pd.to_numeric(df_tmp)
        df_tmp = df_tmp.to_frame() 
        # dos casos 1 para moda, segundo para media.
        if self.impute_type == 'mode':
            mode = df_tmp.mode(axis = 0).values[0]
            self.replace_with = dict(zip(df_tmp.columns, np.reshape(mode, newshape=(-1,))))
        elif self.impute_type == 'mean':
            mean = df_tmp.mean(axis = 0).values[0]
            self.replace_with = dict(zip(df_tmp.columns, np.reshape(mean, newshape=(-1,))))
        return self
    
    def transform(self, X):
        replace_with = {k:{self.missing_value: v} for k, v in self.replace_with.items()}
        return X.replace(to_replace = replace_with).apply(pd.to_numeric, errors = 'ignore')
        
{% endhighlight %}


{% highlight python %}
# Variable que utilizaremos para fijar el mismo 
# random_state a lo largo del problema
random_state = 1
{% endhighlight %}


{% highlight python %}
#Creamos un dataset ficticio con la ayuda de sklearn. 
#En este caso 1000 filas, con 1 variable independiente 
#`x` y 1 dependiente `y`. Al dataset le 
#introducimos un poco de ruido con el parametro noise
X, y = make_regression(n_samples= 1000, n_features=1, n_informative=1, n_targets=1,
               noise=5, random_state=random_state)
{% endhighlight %}
 
La siguiente celda además de llevar, por temas de conveniencia los dos arrays a
un mismo DataFrame, realiza un paso importante: Eleva la variable dependiente al
cuadrado. Lo relevante de este punto, es que explicita el tipo de relación que
deberíamos esperar que `sklearn` detecte al cabo de ajustar el pipeline. Una de
tipo cuadrática. 


{% highlight python %}
df = pd.DataFrame({'X': X.flatten(), 'y': (y**2).flatten()})
{% endhighlight %}
 
A continuación mostramos el gráfico de dispersión de los datos que recién
creamos: `x` e `y`. Esta visualización, además de ilustrar el patrón
anteriormente mencionado muestra, a grandes rasgos, el grado de ruido que tiene
esta relación. 


{% highlight python %}
df.plot(kind='scatter', x = 'X', y = 'y')
{% endhighlight %}




    <matplotlib.axes._subplots.AxesSubplot at 0x1d98f25d1d0>




![png](2018-11-27-un-truquito-de-sklearn...-pipelines%21-2_files/2018-11-27-un-truquito-de-sklearn...-pipelines%21-2_14_1.png)



{% highlight python %}
# Seteamos el mismo random_state para numpy. 
# Para después introducir los carácteres 
# problematicos `'?'` de forma aleatoria.
np.random.seed(random_state)
df.X = df.X.astype(str)
i_choices = np.random.choice(df.index, size = 200)
df.iloc[i_choices, 0] = '?'
{% endhighlight %}
 
## Nuestro Pipeline 
 
Este es el paso clave del post. Aquí la API de `sklearn` tiene como requisito
que entreguemos en una sola lista solo objetos compatibles de `sklearn`. En
mayor detalle, cada elemento de la lista es una tupla que está compuesta por los
nombres que designaremos a cada etapa y con su respectivo objeto de `sklearn`.
Notar que estos objetos que en la siguiente celda estamos llamando son solo
iniciales y su función es netamente actuar como placeholders. 


{% highlight python %}
pipeline = Pipeline([
    ('impute', MyImputer(missing_value = '?')),
    ('transform', FunctionTransformer(np.square, validate=True)),
    ('reg', ElasticNet(random_state=random_state))])
{% endhighlight %}
 
Como último ingrediente relevante es construir el diccionario que `sklearn`
necesita para saber por cuales combinaciones de parámetros debe iterar. Para
ello `sklearn` nos pide que las llaves del diccionario empiecen con los nombres
de cada etapa que definimos en la celda anterior (i.e. `impute`, `transform` o
`reg`) seguido por doble underscore y luego el nombre del parámetro de cada
etapa (i.e. `impute_type` o `func`). Los valores entonces son los parámetros a
recorrer en cada etapa.

Inspeccionando el diccionario, notamos que todos, salvo `log_abs`, son ya sea,
parámetros que ya describimos anteriormente, funciones conocidas de numpy o
metodos importados directamente de `sklearn`. El tratamiento especial en el caso
de `log_abs`, es dado que en el post anterior manifestamos que una
transformación a evaluar era $log(x)$. El problema, es que nuestra variable
independiente `x` contiene valores negativos y $log(x)$ no está definido para
los números negativos. Así, se tuvo que crear la función auxiliar `log_abs` que
tranforma x a valores absolutos y luego utiliza `log`: $log(|x|)$ 


{% highlight python %}
def log_abs(x):
    return np.log(np.abs(x))
{% endhighlight %}


{% highlight python %}
parameters = {'impute__impute_type':  ['mode', 'mean'],
              'transform__func': [np.square, np.exp, np.reciprocal, log_abs],
              'reg': [
    ElasticNet(random_state=random_state),
    Lasso(random_state=random_state), 
    Ridge(random_state=random_state), 
    LinearRegression()
]}
{% endhighlight %}
 
En la siguiente celda es donde finalmente creo se muestra de forma más clara la
magia de sklearn.

Como ya habíamos adelantado, aplicamos `GridsearchCV` directamente al `pipeline`
que definimos. A diferencia de `RandomizedSearchCV` `GridsearchCV` evaluará
exhaustivamente cada combinación posibles del producto cardinal de los sets
definidos en `parameters`. Como check, para la combinación óptima además de
esperar una relación del tipo cuadrática: (`np.square`) debiésemos esperar que
el método de imputación óptimo sea `mean`. La razón es que en el caso de
`impute`,  es muy probable que la moda encontrada para nuestra variable
independiente, no corresponda a un valor que represente a uno de tendencia
central (aspecto deseable para nuestro problema). Esto se da, porque al ser `x`
una variable continua generada aleatoriamente, es practicamente imposible que
encuentre una valor que tenga una frecuencia superior a 1 en dataset. Por lo
tanto, al tener todos los valores la misma frecuencia, en vez de pandas
devolvernos un valor de tendencia central, no le queda más que devolvernos el
array completo de `x`, tomando nuestra clase el primero que encuentra. 


{% highlight python %}
rs_reg = GridSearchCV(pipeline, parameters )
rs_reg = rs_reg.fit(df.X.to_frame(), df.y)
{% endhighlight %}
 
La celda anterior, nos muestra el último gran paso en el uso que le hemos dado a
`Pipelines`. Aquí hacemos uso de la clásica función `fit` de `sklearn`. Como
ven, en una sola linea, logramos ajustar cada una de las combinaciones de
parámetros que definimos anteriormente, identificando, de paso, la combinación
óptima. No me creen? En la siguiente celda, les muestra esta combinación óptima
de parámetros que habiamos mencionado que `GridSearchCV` del `pipeline` debía
encontrar. `mean` y `np.square` para `impute` y `transform`, respectivamente. 


{% highlight python %}
rs_reg.best_params_
{% endhighlight %}




    {'impute__impute_type': 'mean',
     'reg': LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False),
     'transform__func': <ufunc 'square'>}


 
---- 
 
# Conclusión 
 
Para finalizar me gustaría enfatizar una serie de puntos que quise transmitir a
lo largo de estos dos posts:

1. Que podemos extender el ajuste de parámetros que no son parte del modelo,
pero si de etapas previas de forma simple y clara mediante `Pipelines`.
2. Que tenemos la libertad de definir nuestras propias clases de `sklearn` los
cuales nos permiten utilizar los métodos de `sklearn` de forma más generalizada.
3. Que estas clases no necesariamente deben ser limitadas a utilizar funciones
unicamente de `sklearn` o `python` nativo. Específicamente, como parte de una
definición de nuestra clase customizada de `sklearn`, podemos utilizar otras
librerías como `pandas`. Lo cual siempre y cuando es permitido cuando sigamos
los requerimientos solicitados por `sklearn`.

A pesar de lo anterior, no crean que `Pipelines` se restringe solo a esto. Es
más, para este blog temas cómo crear propios estimadores para luego integrarlos
a un `Pipeline`, crear pipelines anidados y definir tratamientos en paralelo de
datasets heterogéneos mediante `FeatureUnions`, fueron aspectos que por tiempo,
no pude considerar para este blog. En este sentido, si no deseas esperar, te
recomiendo  que leas la documentación de [sklearn](https://scikit-
learn.org/stable/modules/classes.html#module-sklearn.pipeline), que tiene varios
ejemplos y guías muy útiles para empezar. 
