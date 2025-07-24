
![wave](media/image/wave.gif)

# **Apresentação**
Olá! Meu nome é Vinícius Fonseca e sou estudante de graduação em **Matemática** na **UEL - Universidade Estadual de Londrina**.  

Desde o meu primeiro "Hello World!" me apaixonei por programação — e desde então, não parei mais.  

Naturalmente, a proximidade entre Matemática e Ciência de Dados me levou a aprender ``Python`` e, com ele, as principais bibliotecas da área:

  - `numpy`
  - `pandas`
  - `matplotlib`
  - `cv2`
  - `YOLO`

As três primeiras foram fundamentais no desenvolvimento do meu projeto ***Uso de LLMs para Detecção de Fake News no Brasil***, no qual utilizei LLMs para gerar embeddings com o Ollama e os classifiquei com um Random Forest Classifier do `scikit-learn`, alcançando métricas ***f1-macro*** entre ***88%*** (pior cenário) e ***99%*** (cenário ideal).

>Confira o projeto aqui **→** [LLM4FakeNews](https://github.com/Viniks07/LLM4FakeNews)   

  
Embora esse projeto tenha trazido importantes aprendizados, minha principal área de interesse continua sendo a **Visão Computacional**.
Ao experimentar diversos exemplos com `cv2` e `YOLO`, percebi que estava adquirindo familiaridade com as bibliotecas, mas sem uma compreensão profunda dos princípios que fundamentam seu funcionamento.

Com esse pensamento, decidi iniciar este novo projeto chamado [MovRecCNN](https://github.com/Viniks07/MovRecCNN):
  
 **Desenvolver um pipeline de detecção e reconhecimento de movimentos sem uso de bibliotecas externas, utilizando apenas:**

- Bibliotecas nativas do ``Python``

- ``numpy`` para manipulação de matrizes

- ``cv2`` ***será utilizado exclusivamente para leitura e escrita de vídeos e imagens***

E por falar em matrizes, está na hora de abordarmos o elemento central de todo este projeto.

# **A Matriz**

Uma das principais representações de imagens para o computador é a matriz onde cada pixel é posicionado por sua linha e coluna na matriz e o cada pixel possui um valor ou uma lista de valores.

A imagem abaixo representa uma matriz em escala de cinza mas pode nos mostrar conceitualmente como funciona o posicionamento de cada pixel e os valores dentro deles.

![Representação Matriz](media/image/matrix_representation.png)

Abaixo temos uma imagem mais explicita de como funcionam esses valores dentro de cada posição.

![Matriz RGB](media/image/gray_matrix.png)

[**Veja como funciona aqui**](https://viniks07.github.io/MovRecCNN/media/html/simulador_de_pixel.html)
