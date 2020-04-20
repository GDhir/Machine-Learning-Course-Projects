
import numpy as np
def input_calc(w, z):
	a1 = np.matmul(w, z)
	return a1

def sigmoid(a, z):
	for i in range(len(a)):
		for j in range(len(a[i])):
			z[i][j] = max(a[i][j], 0)
			z[-1, :] = 1
	
	return z

def sigmadash(a, zdash):
	for i in range(len(a)):
		for j in range(len(a[i])):
			if max(a[i][j], 0)>0:
				zdash[i][j] = 1
			else:
				zdash[i][j] = 0

	return zdash

#zln represents previous z
#an represents a for new layer
#zlp represents output of previous layer
#zln represents output of new layer

def netlayer(wl, zlp, zln):
	an = input_calc(np.transpose(wl),zlp)
	zln = sigmoid(an, zln)
	return an, zln			
