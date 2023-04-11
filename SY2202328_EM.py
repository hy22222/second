import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.neighbors import KernelDensity
import math
np.random.seed(1)


# 定义高斯分布的参数
mean1, std1 = 164, 3
mean2, std2 = 176, 5

# 从两个高斯分布中生成样本
data1 = np.random.normal(mean1, std1, 500)
data2 = np.random.normal(mean2, std2, 1500)
data = np.concatenate((data1, data2), axis=0)
#求参数值
n=[]
for i in data:
	n.append([i])
Y=np.array(n)
# 将数据写入 CSV 文件
df = pd.DataFrame(data, columns=['height'])
df.to_csv('height_data.csv', index=False)

# 绘制数据的直方图
plt.hist(data, bins=20)
plt.xlabel('Height (cm)')
plt.ylabel('Count')
plt.title('Distribution of Heights')
plt.show()


def gauss(y,mu_k,cov_k):
	#返回多元正态随机变量，概率密度函数
	norm = multivariate_normal.pdf(y,mean=mu_k, cov=cov_k)
	return norm

def em(Y,K,iter):
	X = Y
	#初始化参数.a、均值、协方差
	N, D = X.shape
	a = np.ones((K, 1)) / K
	mu = np.array([X[np.random.choice(N)] for _ in range(K)])#两组数据中，Y[i]从2000个数随机选一个数分别作为均值
	cov =np.array([np.eye(D) for _ in range(K)])#两个（2，1，1）协方差阵，各个向量的协方差
	w = np.zeros((N, K))#(2000,2)零阵，两列对应两个k

	for i in range(iter):
		#E-step.求隐变量wik,概率
		p = np.zeros((N, K))
		for k in range(K):
			p[:,k]=a[k]*gauss(X,mu[k],cov[k])
		psum=np.sum(p,axis=1)#将矩阵压缩为1列
		w=p / psum[:, None]

		#M-setp,更新参数a、miu、cov
		sumw = np.sum(w, axis=0)
		#更新a,每一类的占比
		a = sumw / N
		#针对每一类，根据公式形式
		l=[]
		for k in range(K):
			#求均值miu，先求wx的乘积
			w_X = np.multiply(X, w[:, [k]])
			mu[k] = np.sum(w_X, axis=0) / sumw[k]
			#求cov，先求x-mu，再求w*
			X_mu_k = np.subtract(X, mu[k])
			w_X_mu_k = np.multiply(w[:, [k]], X_mu_k)
			#np.transpose矩阵转置，dot矩阵乘法
			cov[k] = np.dot(np.transpose(w_X_mu_k), X_mu_k) / sumw[k]

	return w,a,mu,cov

w,a,miu,cov=em(Y,2,200)
max_index = np.argmax(w, axis=1)

print("初始两类分别为:",0.25,0.75)
print("预测两类分别为:",a[0],a[1])
print("初始均值为:",mean1,mean2)
print("预测均值为:",miu[0],miu[1])
print("初始方差为:",std1*std1,std2*std2)
print("预测方差为:",cov[0],cov[1])

sample1 = np.random.multivariate_normal(mean=miu[0], cov=cov[0], size=500)
sample2 = np.random.multivariate_normal(mean=miu[1], cov=cov[1], size=1500)

n1=[]
for i in data1:
	n1.append([i])
n2=[]
for i in data2:
	n2.append([i])
plt.scatter(n1, sample1, c='red')
plt.scatter(n2, sample2, c='blue')
plt.show()