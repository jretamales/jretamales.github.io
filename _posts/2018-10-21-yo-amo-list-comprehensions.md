---
layout: post
mathjax: true
title: Yo ❤️ list comprehensions
date: 2018-10-21 13:16 +0000
tags: [python]
comments: true
---
 
## Introducción 
 
Una de las cosas que más me gusta de
[Python](https://es.wikipedia.org/wiki/Python) son sus [List
comprehensions](https://docs.python.org/2/tutorial/datastructures.html#list-
comprehensions) o Comprensiones de lista. Creo que que reflejan, una importante
filosofía de Python: privilegiar código sucinto y claro.

Por si recién te estás iniciando en la programación las
[listas](https://docs.python.org/3/tutorial/datastructures.html) son de las
estructuras de datos más básicas que existen. En Python, cada lista tiene
asociado varios métodos y atributos, pero para esta oportunidad lo único que
debes saber es que son un conjunto ordenado de objetos. Por ejemplo: `['hola',
'chao', 1, 2]` 


{% highlight python %}
mi_lista = ['hola', 'chao', 1, 2]
lista = [1,2,4]
{% endhighlight %}
 
## List Comprehensions 
 
Pero que pasa si queremos crear una nueva lista.? El cual tenga mismo numero de
elementos, pero donde cada elemento fue transformado por alguna función
arbitraria? Para los que han programado antes, pensarán que para ello se
necesita crear una lista vacía y luego con un loop for se va agregando uno a uno
cada elemento, por ejemplo: 


{% highlight python %}
lista_transf = []
for elem in lista:
    lista_transf.append(elem**2)

lista_transf
{% endhighlight %}




    [1, 4, 16]


 
No se ustedes, pero a mi me parece que 3 lineas es excesivo para algo tan
simple. Bienvenido List comprehensions... 


{% highlight python %}
lista_transf = [elem**2 for elem in lista]
lista_transf
{% endhighlight %}




    [1, 4, 16]


 
Vieron la línea anterior? Resumimos las tres primera lineas de la anterior celda
en solo una 😊.

Ahora, para entender un poco más lo que esta pasando hay que entender 2
conceptos:
* Que el orden de la sintaxis (o como se escribe), es al revés de como se hace
en el loop for: **Mientras que el orden de los loop for se puede entender como
'Para cada elemento en esta lista haz algo', para la list comprehensions es 'Haz
algo para cada elemento en esta lista'**
* Que el resultado de una list comprehension siempre es un lista.

Con esto --como sabemos que podemos recrear la misma transformación, pero con
otro orden de sintaxis y que el resultado será una lista -- combinarlas en una
sola línea, como lo hicimos con el list comprehension.

En caso que tengas buena memoria de tu curso de matemática, pensarás que la
estructura del list comprehension la has visto en alguna parte y estarás en lo
cierto. Dado que en matemática es usual por ejemplo ver expresiones del tipo
$\{f(x)\space   \forall x \in A\}$, que es básicamente la misma sintaxis
utilizada en list comprehensions. 
 
#### List comprehensions anidada 
 
Pero y que pasa si tenemos una lista anidada del tipo `[[1,2],[3,4]]` y queremos
transformar cada elemento de cada sublista, para obtener algo como `[1, 4, 9,
16]`? Probemos: 


{% highlight python %}
lista = [[1,2],[3,4]]
[elem**2 for elem in lista]
{% endhighlight %}


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-5-27d4013ce44f> in <module>()
          1 lista = [[1,2],[3,4]]
    ----> 2 [elem**2 for elem in lista]
    

    <ipython-input-5-27d4013ce44f> in <listcomp>(.0)
          1 lista = [[1,2],[3,4]]
    ----> 2 [elem**2 for elem in lista]
    

    TypeError: unsupported operand type(s) for ** or pow(): 'list' and 'int'

 
Nuestro primer intento nos entrega un error, donde el mensaje es que estamos
tratando de elevar al cuadrado una lista, lo cual python no sabe cómo
exactamente hacerlo. Pero no se debe perder la esperanza, el truco es
simplemente llegar a cada elemento contenido en las sublistas a través de la
reutilización de la misma sintaxis de list comprehensions. Así: 


{% highlight python %}
[sub_elem**2 for sub_lista in lista for sub_elem in sub_lista]
{% endhighlight %}




    [1, 4, 9, 16]


 
Como ven la sintaxis sigue la misma lógica, un for loop al revés: En vez de
*"Cada sublista en lista y cada elem en sublista haz algo con elem" → "Haz algo
con elem para cada sublista en lista y cada elem en sublista"* 
 
## Dict Comprehensions 
 
Naturalmente, esto también se puede extender a otras estructuras de datos como
los [diccionarios](https://docs.python.org/3/tutorial/datastructures.html#dictio
naries). El cual, en términos muy breves, es una estructura de datos, sin orden,
donde cada llave de la lista está asociada a un valor, de la siguiente forma
`{'llave3': valor3, 'llave1': valor1, 'llave2':valor2,....}` 


{% highlight python %}
mi_dict = {'llave3':3, 'llave1': 1, 'llave2':2 }
{% endhighlight %}
 
Y probemos rápidamente como funcionan las dict comprehensions 


{% highlight python %}
{llave:(val**2) for (llave,val) in mi_dict.items()}
{% endhighlight %}




    {'llave3': 9, 'llave1': 1, 'llave2': 4}


 
Ok... Lo admito, la sintaxis es un poco más complicada que con listas, pero no
demasiado. Sin embargo, si lo analizamos bien, la lógica no cambió. Solo que
para poder recorrer las llaves y valor simultáneamente, necesitamos utilizar el
método `items()`, el cual nos devuelve un listado de `(llave, valor)`. Tomando
cada par `llave, valor` asociado a cada elemento, simplemente reconstruimos el
diccionario, ocupando la sintaxis `llave: valor`. 
