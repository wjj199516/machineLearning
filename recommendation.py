# In[]:
import os
print(os.getcwd())



# In[3]:
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Embedding, Dropout, Dense, Reshape
from keras.layers.merge import Dot, Concatenate
from keras.models import Model, Input
from keras.utils.vis_utils import plot_model


# 进行探索性分析，看看数据集和评分分布长什么样子。
# 计算评分稀疏性，因为所有的推荐系统都是基于大量缺失数据的。
# 预测整个评分表，把缺失数据还原。

# In[4]:

#root_dir = 'F:/2019-notebook/2017_2018_2/python_code/MTrain/MachineLearn/3_ML/26_applications/2_reco/'

ratings = pd.read_csv('ratings.dat', sep = '::',     engine='python',names = ['user_id','movie_id','rating','timestamp'])
n_users = np.max(ratings['user_id'])
n_movies = np.max(ratings['movie_id'])
# 用户数，电影数，评价数
print([n_users, n_movies, len(ratings)])


# In[]:
# 把评价值的统计直方图画出来
plt.hist(ratings['rating'])
plt.show()
print(np.mean(ratings['rating']))


# 进行对用户和内容的建模，使用的是Emdbedding思想。
# Embedding维度为128

# In[5]:

# 建立用户数据模型
k = 128    
model1 = Sequential()
model1.add(Embedding(n_users + 1, k, input_length = 1))
model1.add(Reshape((k,)))

# In[]:
# 建立电影数据模型
model2 = Sequential()
model2.add(Embedding(n_movies + 1, k, input_length = 1))
model2.add(Reshape((k,)))


# In[6]:


model2.input, model2.output

# 通过计算用户和内容的向量乘积，得出评分。

# In[7]:


model = Sequential()

m = Dot(axes=1)([model1.output, model2.output])

model_output = m

model = Model([model1.input, model2.input], model_output)

model.compile(loss = 'mse', optimizer = 'adam')
#model.compile(loss = 'mse', optimizer = 'rmsprop')
#model.compile(loss = 'mse', optimizer = 'adagrad')

#plot_model(model, to_file='reco.png',show_shapes=True)
#exit(0)

# 准备训练数据，代入模型。

# In[8]: 
users = ratings['user_id'].values
movies = ratings['movie_id'].values
X_train = [users, movies]
y_train = ratings['rating'].values


# In[9]:


model.fit(X_train, y_train, batch_size = 500, epochs = 50)


#预测第10号用户对第99号内容的打分。

# In[10]:


i=10
j=99
pred = model.predict([np.array([users[i]]), np.array([movies[j]])])


# In[11]:


print(pred)


#计算模型在训练数据集上的均方差,拟合程度的好坏。

# In[12]:


mse = model.evaluate(x=X_train, y = y_train, batch_size=128)
print(mse)


# 构建深度学习模型。
# 把用户和内容的Embedding合并在一起（concatenate)，
# 作为输入层，然后通过网络模型提取一层层特征，
# 最后用线性变换得出预测评分。

# In[13]:


k = 128
input_1 = Input(shape=(1,))
model1 = Embedding(n_users + 1, k, input_length = 1)(input_1)
model1 = Reshape((k,))(model1)

input_2 = Input(shape=(1,))
model2 = Embedding(n_movies + 1, k, input_length = 1)(input_2)
model2 = Reshape((k,))(model2)


# In[14]:
input_1,input_2, model1,model2


# In[15]:
model = Concatenate()([model1, model2])
model = Dropout(0.2)(model)
model = Dense(k, activation = 'relu')(model)
model = Dropout(0.5)(model)
model = Dense(int(k/4), activation = 'relu')(model)
model = Dropout(0.5)(model)
model = Dense(int(k/16), activation = 'relu')(model)
model = Dropout(0.5)(model)
yhat = Dense(1, activation = 'linear')(model)

model = Model([input_1, input_2], yhat)
model.compile(loss = 'mse', optimizer = "adam")


# 准备好训练数据集，代入模型训练。
# 通过均方差计算模型的拟合程度。

# In[16]:


users = ratings['user_id'].values
movies = ratings['movie_id'].values
label = ratings['rating'].values
X_train = [users, movies]
y_train = label


# In[18]:


model.fit(X_train, y_train, batch_size = 1000, epochs = 50)


# In[19]:


i,j = 10,99
pred = model.predict([np.array([users[i]]), np.array([movies[j]])])

mse = model.evaluate(x=X_train, y=y_train, batch_size=128)
print(mse)
