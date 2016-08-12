""" 
Code still exceeding the maximum recursion depth. There is some problem with my algorithm which I could not figure out. Any help would be greatly appreciated.
I am printing the initial domains and initial assignment
assignment is written to results.txt

The core algorithm from the book is followed but could not able to debug reason for exceeding of recursion depth

"""
from copy import deepcopy
import sys
class csp:
	def __init__(self, l_con_file):
		#initializations
		self.assignment={}
		self.variables={}
		self.neighbours={}
		self.no_of_backtracks=0
		#reading input files
		f = open('adjacent-states', 'r')
		for line in f:
			split_line = line.strip().split()
			self.variables[split_line[0]] = ''
			self.neighbours[split_line[0]] = split_line[1:]
		#print self.variables
		
		self.domains={}#for optimization
		self.values=['A', 'B', 'C', 'D']
		self.legacy_constraints={}
		f = open(l_con_file, 'r')
		for line in f:
			split_line = line.strip().split()
			self.legacy_constraints[split_line[0]] = split_line[1]
		#Domain Initialization for neighbours of legacy crap
		"""
		for k1, v1 in self.legacy_constraints.iteritems():
			self.domains[k1] = v1
		#print "\nDomains1\n"
		"""
		#print "\nDomains1\n"
		for k1, v1 in self.legacy_constraints.iteritems():
			self.domains[k1] = [v1]
			self.assignment[k1] = v1
		for k2,v2 in self.variables.iteritems():
			if ((k2 in self.legacy_constraints)==False):
					self.assignment[k2] = ''
				
		"""
		H = deepcopy(self.domains)
		print "\nInitial Assignment based on legacy constraints\n"
		print self.assignment
		#print self.domains
		for k,v in self.variables.iteritems():
			if (k in H):
				rem=deepcopy(self.values)
				#print rem
				p=H[k]
				#print "\nP\n"
				#print p
				rem.remove(p)
				for N in self.neighbours[k]:
					if ((N in H)==False):
						#print N
						self.domains[N]=deepcopy(rem)
						#print "\nDomains2\n"
						#print self.domains
						#print "\nDomains2\n"
		"""
		self.ass_domains(self.assignment)
		print "\nDomains are being initialized considering legacy constraints\n"
		#print self.domains
		
		n=0
		for it in self.domains:
			n+=1
			print "\n"
			print self.domains[it]
			print "\n"
		print "\nDomain Number: "
		print n
		print "\nAssignment initializations start:\n"
		print self.assignment
		print "\nAssignments initialization finish\n"
	def ass_domains(self,P):
		""" Domain Ordering """
		K=0
		for it in self.domains:
			K+=1
		if (K==50):
			return True
		else:
			H = deepcopy(P)
			for k,v in self.variables.iteritems():
				if (k in H):
					rem=deepcopy(self.values)
					#print rem
					p=H[k]
					#print "\nP\n"
					#print p
					rem = deepcopy(list(set(rem) - set(p)))
					for N in self.neighbours[k]:
						if ((N in H)==False):
							#print N
							self.domains[N]=deepcopy(rem)
			self.result="no solution"
			n=0
			for it in self.domains:
				n+=1
			#print "\nDomain Number: "
			#print n
			#print "\n"
			if (n==50):
				return True
			elif (n==K):
				H = deepcopy(self.domains)
				for k,v in self.variables.iteritems():
					if ((k in H)==False):
						rem=deepcopy(self.values)
						self.domains[k]=deepcopy(rem)
				L=0
				for it in self.domains:
					L+=1
				if (L==50):
					return True
				else:
					self.ass_domains(P)
			else:
				self.ass_domains(P)
	def check_ass_complete(self):
		for k,v in self.assignment.iteritems():
			if (v==''):
				return False
			else:
				return True
	def backtracking_search(self):
		return self.recursive_backtracking()
	def select_unassigned_variable(self):
		#optimization by implementing MRV here
		#for k, v in self.domains.iteritems():
			
		#procedure to give out the unassigned variable
		for k, v in self.variables.iteritems():
			if (v == ''):
				return k
		return
	def ass_domains_replica(self,P,D):
		""" Domain Ordering """
		K=0
		for it in D:
			K+=1
		if (K==50):
			return D
		else:
			H = deepcopy(P)
			for k,v in self.variables.iteritems():
				if (k in H):
					rem=deepcopy(self.values)
					#print rem
					p=H[k]
					#print "\nP\n"
					#print p
					rem = deepcopy(list(set(rem) - set(p)))
					for N in self.neighbours[k]:
						if ((N in H)==False):
							#print N
							D[N]=deepcopy(rem)
			self.result="no solution"
			n=0
			for it in D:
				n+=1
			#print "\nDomain Number: "
			#print n
			#print "\n"
			if (n==50):
				return D
			elif (n==K):
				H = deepcopy(D)
				for k,v in self.variables.iteritems():
					if ((k in H)==False):
						rem=deepcopy(self.values)
						D[k]=deepcopy(rem)
				L=0
				for it in D:
					L+=1
				if (L==50):
					return D
				else:
					self.ass_domains_replica(P,D)
			else:
				self.ass_domains_replica(P,D)
	def consistant(self, var, value):
		P=deepcopy(self.assignment)
		D=self.ass_domains_replica(P,deepcopy(self.domains))
		
		for k,v in D.iteritems():
			#print D
			if(v==[]):
				return False
		return True
	def recursive_backtracking(self): #returns a solution or a failure
		if (self.check_ass_complete()):
			return self.assignment
		if (self.no_of_backtracks>=990):
			return "failure"
		var = deepcopy(self.select_unassigned_variable())
		for value in self.domains[var]:
			if (self.consistant(var,value)):
				self.assignment[var]=value
				self.no_of_backtracks += 1
				#print "\n"
				#print self.no_of_backtracks
				#print "\n"
				self.result = deepcopy(self.recursive_backtracking())
				if (self.result!="failure"):
					return self.result
				else:
					self.assignment[var]=''
		return "failure"

#main
x=csp(sys.argv[1])
if ((x.backtracking_search())!="failure"):
	print "\nResults written to results.txt\n"
	print "No. of backtracks:"+ str(x.no_of_backtracks)
	f=open('results.txt','w')
	for k,v in x.assignment.iteritems():
		s=str(k)+" "+str(v)+"\n"
		f.write(s)
else:
	print "\nfailure\n"
	print "No. of backtracks:"+ str(x.no_of_backtracks)
	f=open('results.txt','w')
	for k,v in x.assignment.iteritems():
		s=str(k)+" "+str(v)+"\n"
		f.write(s)
