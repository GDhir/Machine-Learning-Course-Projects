import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
#import dist2

def dist6(x,c):

	ndata,dimx = x.shape
	ncenters, dimc = c.shape
	xsum = np.sum(x**2,axis = 1)
	xsum = xsum[:,np.newaxis]
	csum = np.sum(c**2,axis = 1)
	csum = csum[:,np.newaxis]
	n2 =  xsum.dot(np.ones([1,ncenters]))+ np.ones([ndata,1]).dot(csum.T)- 2*x.dot(c.T)
	return n2



bodyfat = sio.loadmat('bodyfat_data.mat')
x = bodyfat['X']
y = bodyfat['y']

def gaussian_kernel(u, v):
	sigma = 15
	return -dist6(u,v)/2/sigma**2


def main():
	n = x.shape[0]
	k = np.zeros((n,n))
	print(x[1,:].shape)
	x1 = np.zeros((1,2))
	x2 = np.zeros((1,2))
	lbda = 0.003

	for i in range(n):
		for j in range(n):
			x1[0,:] = x[i,:]
			x2[0,:] = x[j,:]
			k[i,j] = gaussian_kernel(x1, x2)
			#print(x1)

	kO = np.ones((n,n))/n
	Ktilde = k - np.matmul(k, kO) - np.matmul(kO,k) + np.matmul(kO, np.matmul(k, kO))

	ytilde = y-y.mean(axis=0,keepdims = True)

	fx = np.zeros((n,1))
	fx = y.mean(axis = 0, keepdims = True) + np.transpose(np.matmul(np.transpose(ytilde), np.matmul(Ktilde + n*lbda*np.linalg.inv(np.eye(n)), Ktilde)))

	print(y.mean(axis = 0, keepdims = True))
	

	#print(k)




if __name__ == '__main__':
	main()



