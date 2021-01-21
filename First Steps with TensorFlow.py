#!/usr/bin/env python
# coding: utf-8

# In[31]:


("")
000!pip3 install tensorflow
get_ipython().system('pip3 install numpy')
get_ipython().system('pip3 install -U scikit-learn')
get_ipython().system('pip3 install pandas')


# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing


# # Criando um Grafo do Cálculo (Fase de Construção)

# In[2]:


tf.compat.v1.disable_eager_execution()
x = tf.Variable(3,name="x")
y = tf.Variable(4,name="y")
x1 = tf.Variable(1) # Criando um nó que sera adicionado automaticamente ao grafo padrão;
x1.graph is tf.compat.v1.get_default_graph()  # Pega o graph padrão;
f = x*x*y + y + 2


# # Executando o Grafo de Cáclulo (Fase de Execução)

# In[3]:


init = tf.compat.v1.global_variables_initializer() # Inicializa todas as variaveis globais;
with tf.compat.v1.Session() as sess:
    init.run()   # Faz a chamada para inicializar as variaveis globais;
    #x.initializer.run()
    #y.initializer.run()
    result = f.eval()
    #ou:
    #sess.run(x.initializer)
    #sess.run(y.initializer)
    #result = sess.run(f)
    print(x1.eval())
    print(result)


# # Gerenciando Grafos

# In[4]:


#graph = tf.Graph() # Gera um novo Grafo;
#with graph.as_default():   # inicializar com o graph sendo padrão;
 #   x2 = tf.Variable(2)
  #  x2.graph is graph
   # print(x2)

    #tf.reset_default_graph()


# # Criando Grafo de Calculo 2

# In[5]:


w = tf.constant(3)
x = w + 2
y = x + 5
z = y * 3


# # Executando Grafo 2 (Ciclo de Vida de um Valor do nó)

# In[6]:


init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.InteractiveSession()  # Automaticamente se configura como uma sessão padrão;
init.run()
y_val,z_val = sess.run([y,z])  # para calcular y e z, sem executar w e x duas vezes;
print(y_val)
print(z_val)
#print(y.eval())
#print(z.eval())
sess.close()  # Precisa ser usado para encerrar a sessão manualmente;


# # Regressão Linear com o TensorFlow

# In[7]:


housing = fetch_california_housing()  # Datasheet com informações imobiliarias da California;
m,n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m,1)),housing.data]


# In[8]:


x = tf.constant(housing_data_plus_bias,dtype = tf.float32,name="X")  # nó constante que segura os dados
y = tf.constant(housing.target.reshape(-1,1),dtype = tf.float32,name="Y") # nó constante que segura o alvo (target)
T = tf.transpose(x)
theta = tf.matmul(tf.matmul(tf.compat.v1.matrix_inverse(tf.matmul(T,x)),T),y) # Minimos quadrados (maltmul = multi.de matrizes)

with tf.compat.v1.Session() as sess:
    theta_value = theta.eval()  # eval() realiza a execução do cálculo e retorna o valor final.
    print(theta_value) # Valor do Theta usando Método dos Minimos qudrados (0 = (X^T.X)^-1.x^T.y)


# # Fornecendo Dados ao Algoritmo de Treinamento

# In[9]:


# Usando a função placeholder();
A = tf.compat.v1.placeholder(tf.float32,shape = (None,3)) #None = qualquer tamanho
B = A + 5
with tf.compat.v1.Session() as sess:
    B_val_1 = B.eval(feed_dict = {A:[[1,2,3]]})  # passando o valor para o placeholder A;deve-se usar a função feed_dict;
    B_val_2 = B.eval(feed_dict = {A:[[4,5,6],[7,8,9]]})

print(B_val_1)
print(B_val_2)


# # Salvar e Restaurar Modelos

# In[ ]:




