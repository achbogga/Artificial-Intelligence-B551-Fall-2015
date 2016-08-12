'''
Author: Achyut Sarma Boggaram (IU username: achbogga)
Rameses.py
python rameses.py 3 .x......x
Python Version 3.3
Need to change print statements to run it in python 2.7

Each state is represented as size squared list of x's and .'s


function written for Task3: Generate minimax values for each node in a game tree
structure written for Task1: The representation of states as a game tree
function for Task2: The generation of game tree for a given board configuration

Problem Unexpected behaviour while generating successor nodes - > No new instances created (Could not figure out why!) -> Solved

Problem The leaves are not updated of their minimax Values -> This is weird!!! 

#(* Initial call for maximizing player *)
minimax(origin, depth, TRUE)    

'''
import sys
import math
from copy import deepcopy
index=0

class GameTree:
    def __init__(self, origin = None, children = [], parent = None, Data = [], depth = 0):
        self.origin = origin
        self.children = children
        self.parent = parent
        self.data = Data
        self.depth = depth
        self.minimaxValue=-80000
        #self.index = index
        #Successors Generation not working if called as a constructor
        #self.GenSuccessorsOneLevel(self)
        
    def minimax(self, node, depth, Max):
        if ((depth == 0) or (self.IfLeaf(node))):
            return (self.StaticEval(node))
        if Max:
            bestValue = -10000 #-10000 means -infinity
            for child in self.children:
                val = child.minimax(child, depth -1, False)
                bestValue = max(bestValue, val)
            return bestValue
        else:
            bestValue = +10000 #+10000 means +infinity
            for child in self.children:
                val = child.minimax(child, depth - 1, True)
                bestValue = min(bestValue, val)
            return bestValue
        

    #Static Eval
    #Heuristic returns 0 for any losing state and returns count of unfilled rows, columns and diagonals.
    def DispTree(self):
        if self.IfLeaf (self):
            print ("Data and Depth of the Node")
            print (self.data)
            print (self.depth)
        else:
            for child in self.children:
                child.DispTree()
    def StaticEval(self, node):
        #horizontal
        size=int(math.sqrt(len(node.data)))
        RowValue=0
        ColValue=0
        DiagValue=0
        for y in range(size):
            losing = []
            for x in range(size):
                if self.data[x*y] == 'x':
                    losing.append((x,y))
            if len(losing) == size:
                return 0
            else:
                RowValue += 1
        # vertical
        for y in range(size):
            losing = []
            for x in range(size):
                if self.data[x*y] == 'x':
                    losing.append((x,y))
            if len(losing) == size:
                return 0
            else:
                ColValue += 1
        # diagonal1
        losing = []
        for y in range(size):
            x=y
            if self.data[x*y] == 'x':
                losing.append((x,y))
        if len(losing) == size:
            return 0
        else:
            DiagValue += 1
        # diagonal2
        losing = []
        for x in range(size):
            y=x
            if self.data[x*y] == 'x':
                losing.append((x,y))
        if len(losing) == size:
            return 0
        else:
            DiagValue += 1
        # default Evaluation if not a goal state
        return (DiagValue+RowValue+ColValue)
    def IfLeaf(self, node):
        if(len(node.children)==0):
            return True
        else:
            return False
    def GenSuccessorsRec(self,node):#returns the node after generating all its successors
        #global index;
        size=int(math.sqrt(len(node.data)))
        emptyPositions=[]
        for l in range(int(len(node.data))):
            if node.data[l] == '.':
                emptyPositions.append(l)
        if self.StaticEval(node)==0:
            node.children=[]
            return node
        else:
            node.depth += 1 #increases every level
            count=0
            new = []
            for i in range(len(emptyPositions)):
                N=deepcopy(GameTree(node.origin, [], node, node.data, (node.depth-1)))
                new.append(N)
                new[count].data[emptyPositions[i]]='x'
                #print(emptyPositions)
                #print(new[count].data)#New instances are not created!!! -> This is ridiculous!!!  
                count += 1
            node.children=deepcopy(new)
            for child in node.children:
                child.GenSuccessorsRec(child)
'''---------------------------------MAIN PROGRAM------------------------------------------------------------ '''
#main program
#sys.setrecursionlimit(5000) - > Not Required
(size, State, TimeLimit) = sys.argv[1:]
Data=[]
for c in State:
    Data.append(c)
    
root = GameTree(None, [], None, Data, 1)
root.GenSuccessorsRec(root)
def MiniMax (node):
    if (node.IfLeaf(node)):
        #print("control reaches here")
        return None
    else:
        if ((node.depth%2)==0):
            #print("L1")
            node.minimaxValue = deepcopy(node.minimax(deepcopy(node), deepcopy(node.depth), False))
        else:
            #print("L2")
            node.minimaxValue = deepcopy(node.minimax(deepcopy(node), deepcopy(node.depth), True))
        for child in deepcopy(node.children):
            #print("I am for Loop")
            MiniMax(child)
MiniMax(root)
#print (root.depth)
#print ("MiniMax Value of Root: " + root.minimaxValue + "\n")
Opt=[]
def PathFinder(node):
    if (node.IfLeaf(node) and (node.minimaxValue==0)):
        while(node.depth==2):
            node=deepcopy(node.parent)
        #print(node.data)
        #print ("Control Reached here")
        #print(deepcopy(node.data))
        Opt.append(deepcopy(node.data))
    else:
        for child in node.children:
            if (child.minimaxValue==0):
                #Opt.append(deepcopy(node.data))
                PathFinder(child)
        #print("end of FOr here")
'''if (root.IfLeaf(root)):
    print("")
else:
    for child in root.children:
        #print(child.minimaxValue)
        if(child.minimaxValue==0):
            Opt.append(child)'''
PathFinder(root)
#print(Opt)
#root.DispTree()
print "Given Below are the recommended moves";
for option in Opt:
    #print(option)
    print ''.join(option)

''' The leaves are not updated of their minimax Values -> This is weird!!! '''
def leaves(node):
    if (node.IfLeaf(node)):
        #print(node)
        print(node.minimaxValue)
    else:
        for child in deepcopy(node.children):
            leaves(child)
#leaves(root)
