#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

X=np.array([[1,1],[1.1,1.1],[1.2,1.2],
            [2,2],[2.1,2.1],[2.2,2.2]])

y=[1,1,1,
  0,0,0]
kmeans = KMeans(n_clusters=2,random_state=0).fit(X)
print("集群中心的座標:",kmeans.cluster_centers_)
print("預測:",kmeans.predict(X))
print("實際:",y)
print("預測[1,1],[2.3,2.1]:",kmeans.predict([[1,1],[2.3,2.1]]))

plt.axis([0,3,0,3])
plt.plot(X[:3,0],X[:3,1],'yx')
plt.plot(X[3:,0],X[3:,1],'g.')
plt.plot(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],'ro')
plt.xticks(())
plt.yticks(())
plt.show()


# In[18]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import metrics

iris = datasets.load_iris()

iris_X_train ,iris_X_test ,iris_y_train, iris_y_test =train_test_split(iris.data,iris.target,test_size=0.2)

kmeans = KMeans(n_clusters = 3)
kmeans_fit = kmeans.fit(iris_X_train)

print("實際",iris_y_train)
print("預測",kmeans_fit.labels_)
iris_y_train[iris_y_train==1]=11
iris_y_train[iris_y_train==0]=1
iris_y_train[iris_y_train==11]=0
print("調整",iris_y_train)
score=metrics.accuracy_score(iris_y_train,kmeans.predict(iris_X_train))
print("準確率:{0:f}".format(score))


# In[16]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import metrics

iris = datasets.load_iris()

iris_X_train ,iris_X_test ,iris_y_train, iris_y_test =train_test_split(iris.data,iris.target,test_size=0.2)
kmeans = KMeans(n_clusters = 3)
kmeans.fit(iris_X_train)
y_predict = kmeans.predict(iris_X_train)

x1=iris_X_train[:,0]
y1=iris_X_train[:,1]
plt.scatter(x1,y1,c=y_predict,cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0],centers[:,1],c='black',s=200,alpha=0.5);
plt.show()


# In[20]:


#Decision Trees 決策樹
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Parameters
n_classes = 3
plot_colors = "bry"
plot_step = 0.02

# Load data
iris = load_iris()

for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):

    # We only take the two corresponding features
    X = iris.data[:, pair]
    y = iris.target
    # Train
    clf = DecisionTreeClassifier().fit(X, y)

    # Plot the decision boundary
    plt.subplot(2, 3, pairidx + 1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))


    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) #np.c_ 串接兩個list,np.ravel將矩陣變為一維

    Z = Z.reshape(xx.shape)


    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])
    plt.axis("tight")


    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                    cmap=plt.cm.Paired)

    plt.axis("tight")

plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend()
plt.show()


# In[ ]:


k-means算法特點在於：同一聚類的簇內的對象相似度較高；而不同聚類的簇內的對象相似度較小。

Decision trees（決策樹）是一種過程直覺單純、執行效率也相當高的監督式機器學習模型，
適用於classification及regression資料類型的預測，與其它的ML模型比較起來，執行速度是它的一大優勢。

而Logistic Regression與Support Vector Machines就好像黑箱一樣，我們很難去預測或理解它們內部複雜的運作細節。
而且Decision trees有提供指令讓我們實際的模擬並繪出從根部、各枝葉到最終節點的決策過程。

