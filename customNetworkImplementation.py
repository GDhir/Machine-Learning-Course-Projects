import numpy as np
import scipy as sio
from scipy import loadmat
import random
import input_calculation
import copy
import os

os.chdir('.')
bodyfat = sio.loadmat('bodyfat_data.mat')
x = bodyfat['X']
y = bodyfat['y']
n = x.shape[0]
Ntrain = 150
xTrain = x[:150,:]
yTrain = y[:150]
xTest = x[150:,:]
yTest = y[150:]
z0pre = np.zeros((150,3))
z0pre[:150, 0:1] = xTrain[:150, 0:1]
z0 = np.transpose(z0pre)
z0[2,:] = 1
np.random.seed(90)
w1 = np.random.normal(0.2, 0.3, (3, 64))
z1 = np.zeros((65,150))
np.random.seed(176)
w2 = np.random.normal(0.2, 0.3, (65, 16))
z2 = np.zeros((17,150))
np.random.seed(1102)
w3 = np.random.normal(0.2, 0.3, (17, 1))
z3 = np.zeros((2,150))
tol = 0.01
error = 10
MSE = 10
ll = 0

while MSE > tol:
	[a1, z1] = input_calculation.netlayer(w1, z0, z1)
	[a2, z2] = input_calculation.netlayer(w2, z1, z2)
	[a3, z3] = input_calculation.netlayer(w3, z2, z3)
	delta3pre = -2*(np.transpose(yTrain) - z3[0,:])
	delta3 = np.transpose(delta3pre)
	zdash3 = np.zeros((16, 150))
	zdash3 = input_calculation.sigmadash(a2, zdash3)
	delta2 = np.multiply(np.matmul(delta3, np.transpose(w3[0:16, :])), np.transpose(zdash3))
	zdash2 = np.zeros((64, 150))
	zdash2 = input_calculation.sigmadash(a1,zdash2)
	delta1 = np.zeros((150, 64))
	for k in range(16):
		delta1 = delta1 + np.matmul(np.reshape(delta2[:,k], (150, 1)), np.transpose(np.reshape(w2[0:64,k], (64,1))))
		der1 = np.zeros((150, 64, 2))
		der2 = np.zeros((150, 16, 64))
		der3 = np.zeros((150, 1, 16))
	
	for nn in range(150):
		for j in range(64):
			for i in range(2):
				der1[nn, j, i] = delta1[nn, j]*z0[i, nn]
	
	for nn in range(150):
		for j in range(16):
			for i in range(64):
				der2[nn, j, i] = delta2[nn, j]*z1[i, nn]
	
	for nn in range(150):
		for j in range(1):
			for i in range(16):
				der3[nn, j, i] = delta3[nn, j]*z2[i, nn]
		
	w1temp = copy.deepcopy(w1)
	w2temp = copy.deepcopy(w2)
	w3temp = copy.deepcopy(w3)

	gamma = 1e-7
	for j in range(64):
		for i in range(2):
			t = 0
			for nn in range(150):
				t = t + der1[nn, j, i]
		
			w1[i,j] = w1[i,j] - gamma*t/150
		
	for j in range(16):
		for i in range(64):
			t = 0
			for nn in range(150):
				t = t + der2[nn, j, i]
			
			w2[i,j] = w2[i,j] - gamma*t/150
	
	for j in range(1):
		for i in range(16):
			t = 0
			for nn in range(150):
				t = t + der3[nn, j, i]
			w3[i,j] = w3[i,j] - gamma*t/150
		
	error1 = np.sum(np.square(w1-w1temp))
	error2 = np.sum(np.square(w2-w2temp))
	error3 = np.sum(np.square(w3-w3temp))
		
	error = error1 + error2 + error3
		
	MSE = np.sum(np.square(a3 - np.transpose(yTrain)))/150
	print(MSE)
	ll = ll + 1
	if ll > 1000:
		break

z0newpre = np.zeros((98,3))
z0newpre[:98, 0:1] = xTest[:98, 0:1]
z0new = np.transpose(z0newpre)
z0new[2,:] = 1
z1new = np.zeros((65,98))
z2new = np.zeros((17,98))
z3new = np.zeros((2,98))
[a1new, z1new] = input_calculation.netlayer(w1, z0new, z1new)
[a2new, z2new] = input_calculation.netlayer(w2, z1new, z2new)
[a3new, z3new] = input_calculation.netlayer(w3, z2new, z3new)
MSEtest = np.sum(np.square(a3new - np.transpose(yTest)))/98
print(MSEtest)