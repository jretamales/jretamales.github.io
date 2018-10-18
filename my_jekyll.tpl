{% extends 'markdown.tpl' %}



{% block in_prompt %}
{% endblock in_prompt %}

{% block input %}
{{ '{% highlight python %}' }}
{{ cell.source }}
{{ '{% endhighlight %}' }}
{% endblock input %}

{% block markdowncell scoped %} 
{{ cell.source | wrap_text(80) }} 
{% endblock markdowncell %} 

{% block headingcell scoped %}
{{ '#' * cell.level }} {{ cell.source | replace('\n', ' ') }}
{% endblock headingcell %}