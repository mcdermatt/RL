import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

#throwing errors

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor #uncomment to run on GPU

N = 64 #Batch size
D_in = 1000 #input dimension
H = 100 # hidden dimension
D_out = 10 #output dimension

#make random tensor to hold input and output
#   set requires_grad = False so we don't compute gradients with respect to these 
#	during back propogation
x = Variable(torch.randn(N, D_in).type(dtype), requires_grad = False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad = False)

#create weight tensors and make them variables
#	we want to compute the grad of these
w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad = True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad = True)

learning_rate = 1e-6
for t in range(500):

	#Forward Pass: compute predicted y using operations on variables 
	#	mm -> matrix multiply
	#	clamp -> clamps elements in input array from min to max
	y_pred = x.mm(w1).clamp(min=0).mm(w2)
	#compute loss 
	loss = (y_pred - y).pow(2).sum()
	# print(t, loss.data[0]) #throwing error: invalid index of a 0-dim tensor. Use `tensor.item()` in Python or `tensor.item<T>()` in C++ to convert a 0-dim tensor to a number 
	print(t, loss.data)

	#Manually zero gradients before running backward pass
	w1.grad.data.zero_()
	w2.grad.data.zero_()

	#Backward Pass: computes gradient of loss with respect to all variables with requires_grad = True
	#	after this w1.grad and w2.grad will be variables holding the grad of loss respectively
	loss.backward()

	#update weights using gradient descent
	#	w1.data and w2.data are Tensors
	#	w1.grad and w2.grad are Variables, w1.grad.data and w2.grad.data are tensors
	w1.data -= learning_rate * w1.grad.data
	w2.data -= learning_rate * w2.grad.data

#practicing Backpropogation
# x = Variable(torch.ones(2, 2) * 2, requires_grad=True)
# z = 2 * (x * x) + 5 * x
# z.backward(torch.ones(2, 2))
# print(x.grad)