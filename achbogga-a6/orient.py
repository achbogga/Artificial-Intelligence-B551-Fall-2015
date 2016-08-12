#Achyut Sarma Boggaram @achbogga a6 B551 fall-15
#Please email me if you wish to anything else about the code

	#Nearest k neighbours knn()
	#######Design Decisions####################
	#Distance metric used is Euclidean distance
	#Standardization of each dimension over the whole dataset done
	
	#Neural nets with one hidden layer nnet()->running out of memory for the complete training data on my system -> I am sticking with k-nearest neighbour algorithm as my best model and k value is optimum for 13.
	#######Design Decisions####################
	#Orientations are taken as continuous floating point variables rather than classes
	#Normalization of each dimension over the whole dataset done
	#Newton BFGS optimization used with the help of scipy and numpy

import sys

def knn():	
	#Nearest k neighbours
	import statistics
	import math
	import operator
	from copy import deepcopy
	
	orients=["0","90","180","270"]
	
	class knn:
	    def __init__(self, Training_file, Test_file, classifier, param):
			self.confusion_matrix=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
			class photo:
				def __init__(self, id, o, v):
					self.id=id
					self.right_orient=o
					self.vector=v
			
			
			
			#reading the data for training
			self.Dataset_training=[]
			file = open(Training_file, 'r')
			
			#standardization of the vectors
			L=[]
			
			for line in file:
				line=line.split()
				vector=line[2:]
				
				l=[]
				for n in vector:
					l.append(float(n))
					#print "l:"
					#print l
				L.append(l)
			print "no. of training images read:"
			print len(L)
			print "\n"
			L=zip(*L)
			for t in L:
				m=statistics.mean(t)
				#print m
				M=statistics.pstdev(t)
				if (M==0):
					M=0.001
				for k in t:
					k=((k-m)/M)
			L=zip(*L)
			self.L=L
			#print "L: "
			#print L
			i=0
			file = open(Training_file, 'r')
			for line in file:
				line=line.split()
				#print "\ntest-sample:"
				#print line[0]
				p1=photo(line[0],line[1],self.L[i])
				#p1=[line[0],line[1],L[i]]
				#print p1
				self.Dataset_training.append(p1)
				i+=1
			print "\nStandardization done for training data set..\n"
			#print self.Dataset_training
				#print self.photos
			#k nearest neighbours
			
			#k is initially set to 3 and majority vote being 2
			
			self.k=int(param)
			if ((self.k%2)==0):
				self.vote=(self.k/2)+1
			else:
				self.vote=(self.k+1)/2
			
			#reading the data from test dataset
			self.Dataset_test=[]
			file = open(Test_file, 'r')
			
				#standardization of the vectors
			Lt=[]
			for line in file:
				line=line.split()
				vector=line[2:]
				
				l=[]
				for n in vector:
					l.append(float(n))
					#print "l:"
					#print l
				Lt.append(l)
			print "no. of test images read:"
			print len(Lt)
			print "\n"
			Lt=zip(*Lt)
			for t in Lt:
				m=statistics.mean(t)
				#print m
				M=statistics.pstdev(t)
				if (M==0):
					M=0.001
				for k in t:
					k=((k-m)/M)
			Lt=zip(*Lt)
			self.Lt=Lt
			#print "L: "
			#print L
			i=0
			file = open(Test_file, 'r')
			for line in file:
				line=line.split()
				#print "\ntest-sample:"
				#print line[0]
				self.Dataset_test.append(photo(line[0],line[1],self.Lt[i]))
				#p2=[line[0],line[1],L[i]]
				#Dataset_test.append(p2)
				i+=1
			print "\nStandardization done for test data set..\n"
			
			#Distance calculation is done by using euclidean metric
			#Classification of the whole test data
			for tes_im in self.Lt:
				D=[]
				for train_im in self.L:
					D.append(math.sqrt(sum(map(lambda x: x ** 2,map(operator.sub, tes_im, train_im)))))
				d=deepcopy(D)
				d.sort()
				voting=[0,0,0,0]
				#print "index: "
				#print Dataset_training
				#print "\n"
				for i in range(self.k):
					if (self.Dataset_training[D.index(d[i])].right_orient=="0"):
						voting[0]+=1
					elif (self.Dataset_training[D.index(d[i])].right_orient=="90"):
						voting[1]+=1
					elif (self.Dataset_training[D.index(d[i])].right_orient=="180"):
						voting[2]+=1
					elif (self.Dataset_training[D.index(d[i])].right_orient=="270"):
						voting[3]+=1
					#print "\nVoting vector: "
					#print voting
					#print "\nmax:\n"
					#print voting.index(max(voting))
				self.Dataset_test[self.Lt.index(tes_im)].est_orient=orients[voting.index(max(voting))]
						
	
	
	
	p=knn(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
	out=sys.argv[4]+"nn_output.txt"
	file = open(out,"w")
	for im in p.Dataset_test:
		#st2= ' '.join(map(str, (map(int, im.vector))))
		p.confusion_matrix[orients.index(im.right_orient)][orients.index(im.est_orient)]+=1
		st=im.id+' '+im.est_orient+' '+'\n'
		file.write(st)
	
	print "\nConfusion Matrix: \n"
	for row in p.confusion_matrix:
		print row
	correct=0.0
	wrong=0.0
	total=0.0
	for i in range(4):
		for j in range(4):
			total+=p.confusion_matrix[i][j]
			if (i==j):
				correct+=p.confusion_matrix[i][j]
			else:
				wrong+=p.confusion_matrix[i][j]
	
	p.classification_accuracy=100*(correct/total)
	
	print "\nClassification with k-nearest K("
	print p.k
	print ") neighbours done...\n"
	
	print "\nClassification accuracy percentage: \n"
	
	print p.classification_accuracy 
	
	
	

	
def nnet():
	###code skeleton courtesy @stephenwelch youtube
	#Neural Networks
	import numpy as np
	
	file = open(sys.argv[1], 'r')
	x=[]
	y=[]
	for line in file:
		line=line.split()
		y.append(float(line[1]))
		x.append(line[2:])
	
	# X = image vectors, y = right orientations
	X = np.array(x, dtype=float)
	y = np.array(y, dtype=float)
	
	# Normalize
	X = X/np.amax(X, axis=0)
	y = y/360 #Max right orientaion is 360
	
	
	
	class Neural_Network(object):
	    def __init__(self):        
	        self.inputLayerSize = len(X[0])
	        self.outputLayerSize = 1
	        self.hiddenLayerSize = int(sys.argv[4])
	        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
	        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
	        
	    def forward(self, X):
	        self.z2 = np.dot(X, self.W1)
	        self.a2 = self.sigmoid(self.z2)
	        self.z3 = np.dot(self.a2, self.W2)
	        yHat = self.sigmoid(self.z3) 
	        return yHat
	        
	    def sigmoid(self, z):
	        return 1/(1+np.exp(-z))
	    
	    def sigmoidPrime(self,z):
	        return np.exp(-z)/((1+np.exp(-z))**2)
	    
	    def costFunction(self, X, y):
	        self.yHat = self.forward(X)
	        J = 0.5*sum((y-self.yHat)**2)
	        return J
	        
	    def costFunctionPrime(self, X, y):
	        self.yHat = self.forward(X)
	        
	        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
	        dJdW2 = np.dot(self.a2.T, delta3)
	        
	        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
	        dJdW1 = np.dot(X.T, delta2)  
	        
	        return dJdW1, dJdW2
	    
	    def getParams(self):
	        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
	        return params
	    
	    def setParams(self, params):
	        W1_start = 0
	        W1_end = self.hiddenLayerSize * self.inputLayerSize
	        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
	        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
	        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
	        
	    def computeGradients(self, X, y):
	        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
	        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))
	
	def computeNumericalGradient(N, X, y):
	        paramsInitial = N.getParams()
	        numgrad = np.zeros(paramsInitial.shape)
	        perturb = np.zeros(paramsInitial.shape)
	        e = 1e-4
	
	        for p in range(len(paramsInitial)):
	            perturb[p] = e
	            N.setParams(paramsInitial + perturb)
	            loss2 = N.costFunction(X, y)
	            
	            N.setParams(paramsInitial - perturb)
	            loss1 = N.costFunction(X, y)
	
	            numgrad[p] = (loss2 - loss1) / (2*e)
	
	            perturb[p] = 0
	            
	        N.setParams(paramsInitial)
	
	        return numgrad 
	   
	
	from scipy import optimize
	
	
	class trainer(object):
	    def __init__(self, N):
	        self.N = N
	        
	    def callbackF(self, params):
	        self.N.setParams(params)
	        self.J.append(self.N.costFunction(self.X, self.y))   
	        
	    def costFunctionWrapper(self, params, X, y):
	        self.N.setParams(params)
	        cost = self.N.costFunction(X, y)
	        grad = self.N.computeGradients(X,y)
	        return cost, grad
	        
	    def train(self, X, y):
	        self.X = X
	        self.y = y
	
	        self.J = []
	        
	        params0 = self.N.getParams()
	
	        options = {'maxiter': 200, 'disp' : True}
	        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
	                                 args=(X, y), options=options, callback=self.callbackF)
	
	        self.N.setParams(_res.x)
	        self.optimizationResults = _res
	        
	N=Neural_Network()
	T=trainer(N)
	T.train(X,y)
	
	file = open(sys.argv[2], 'r')
	x=[]
	y=[]
	ids=[]
	for line in file:
		line=line.split()
		ids.append(line[0])
		y.append(float(line[1]))
		x.append(line[2:])
	
	# X = image vectors, y = right orientations
	X = np.array(x, dtype=float)
	
	# Normalize
	X = X/np.amax(X, axis=0)
	
	Res=N.forward(X)
	
	f2 = open("nnet_output.txt", 'w')
	for id in ids:
		im=id+Res[ids.index(id)]+"\n"
		f2.write(im)

	
def best():
	f3= open("test-data.txt", 'r')
	f = open(sys.argv[4], 'r')#model_file is 13nn_output.txt
	r=0
	w=0
	rt=[]
	for l in f3:
		l=l.split()
		rt.append(l[0])
	i=0
	for line in f:
		line=line.split()
		
		ryt=rt[i]
		i+=1
		est=line[0]
		if (est==ryt):
			r+=1
		else:
			w+=1
	ca=(float(r)*100)/float(r+w)
	print "output data written to best_output.txt"
	print "\nclassification accuracy %: "
	print ca
	print  "\n"
	import shutil
	shutil.copyfile(sys.argv[4],"best_output.txt")

#main

if (sys.argv[3]=='knn'):
	knn()
elif (sys.argv[3]=='nnet'):
	nnet()
elif (sys.argv[3]=='best'):
	best()

