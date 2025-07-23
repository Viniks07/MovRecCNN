![wave](media/wave.gif)

# Apresentação
Olá! Meu nome é Vinícius Fonseca e sou estudante de graduação em <span style ="color:#4169E1">Matemática</span> na <span style="color:#228B22">UEL - Universidade Estadual de Londrina</span>.  

Desde o meu primeiro "Hello World!" me apaixonei por programação — e desde então, não parei mais.  

Naturalmente, a proximidade entre Matemática e Ciência de Dados me levou a aprender <span style = "color:#FFD700">Python</span> e, com ele, as principais bibliotecas da área:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `cv2`
  - `YOLO`

As três primeiras foram fundamentais no desenvolvimento do meu projeto __*Uso de LLMs para Detecção de Fake News no Brasil*__, no qual utilizei LLMs para gerar embeddings com o Ollama e os classifiquei com um Random Forest Classifier do `scikit-learn`, alcançando métricas __*f1-macro*__ entre __*88%*__ (pior cenário) e __*99%*__ (cenário ideal).

>Confira o projeto aqui → [LLM4FakeNews](https://github.com/Viniks07/LLM4FakeNews) 
  
Apesar da conclusão e aprendizado desse projeto, meu maior interesse está em __Visão Computacional__. Depois de explorar diversos exemplos com __CV2__ e __YOLO__, percebi que estava aprendendo as ferramentas, mas não os fundamentos por trás delas.

Com esse pensamento, decidi iniciar este novo projeto:
  
### Desenvolver um pipeline de detecção e reconhecimento de movimentos sem uso de bibliotecas externas, utilizando apenas:
- Bibliotecas nativas do Python

- NumPy para manipulação de matrizes

- <span style="color:red;"><strong><em>CV2 apenas para leitura e escrita de vídeos/imagens</em></strong></span>

E por falar em matrizes, está na hora de abordarmos o elemento central de todo este projeto.