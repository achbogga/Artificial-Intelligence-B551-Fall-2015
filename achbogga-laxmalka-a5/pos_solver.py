###################################
# CS B551 Fall 2015, Assignment #5
#
# Laxmi Bhavani Malkareddy : laxmalka
#
# (Based on skeleton code by D. Crandall)
#
#
####
# # ********************************************************REPORT*********************************************************
# The goal of this class is to assign parts of speech tag to english sentences. An optimum solution refers to maximize the  
# accuracy of the parts of speech tagger.
# The program learns from the traning file and applies various algithms to determine the POS for sentences in test file 
#      --------------------------------------------------ALGORITHM-------------------------------------------------------
# 1. Learning: performed by train() 
#              -this function keeps track of all the condition probabilities and transition probabilities of the words and 
#              corresponding parts of speech tags encountered in the traning database. This involves estimating parameters like 
#              which POS tag is likely to follow a particular POS tag[P(si+1/si)], the most probable tag for a particular word[P(w/s)]. 
#
# 2. Naive Inference: performed by naive() 
#              -this functon calculates all the possible P(si/w) values for a given word, finds the maximum, and assigned corresponding tag
#              p(si/w) = p(w and s)/p(w)
#
# 3. Sampling: performed by mcmc()
#              -In order to give a headstart to good sampling, the initial sample is generated from naive inference and fed into the algorithm.
#              Here, 5 samples are generated from the possible samples using Gibbs Sampling technique
#              The samples are collected such that the samples which are most likely accurate are generated. This is achieved by 
#              applying the concept of rolling a biased dice randomly and assigning the samples accordingly
#
# 4. Approximate Max Mariginal Inference: performed by max_marginal()
#              -Similar to gibbs sample. The same code is run for a large number of times and the sample that occurs in the end is considered as output
#
# 5. Viterbi: performed by viterbi()
#              Based on estimated probabilities and transition probabilities, the best once are use to determine tags at next state.
# 6. Best: performed by best()
#              Upon observing the trends of results from above algorithms, a combination of naive and max_marginal algorithms is used tp perform tagging.
#              The naive inference output is fed as initial sample to max_marginal that iterates over a set number of samples and generate the output.
#
#
#		-------------------------------------------END OF THE ALGORITHM---------------------------------------------------
#========================================================================================================================
#Limitations: Takes considerably large time to eecute for large datasets
#
#
#========================================================================================================================
#	RUNNING THE PROGRAM: python label.py bc.train bc.test.tiny
#	MIN RUNTIME: ~10 seconds for a tiny test set
# AVERAGE RUNTIME: ~10 min for huge test set containing upto 2000 sentences
#	Sample Output for a game:
'''
[laxmalka@silo postag]$ python label.py bc.train bc.test.tiny
Learning model...
Loading test data...
Testing classifiers...
                          : poet twisted again and  nick's knuckles scraped on   the  ai  r  tank ,    ripping off  the  skin .
 0. Ground truth (-143.00): noun verb    adv   conj noun   noun     verb    adp  det  no  un noun .    verb    prt  det  noun .
        1. Naive (-141.54): noun verb    adv   conj adj    noun     verb    adp  det  no  un noun .    verb    prt  det  noun .
      2. Sampler (-142.93): noun verb    adv   conj adv    verb     verb    adp  det  no  un noun .    verb    prt  det  noun .
                 (-142.95): noun verb    adv   conj adv    verb     verb    adp  det  no  un noun .    verb    adp  det  noun .
                 (-142.95): noun verb    adv   conj adv    verb     verb    adp  det  no  un noun .    verb    adp  det  noun .
                 (-142.95): noun verb    adv   conj adv    verb     verb    adp  det  no  un noun .    verb    adp  det  noun .
                 (-144.50): noun verb    adv   conj adv    noun     verb    adp  det  no  un noun .    verb    prt  det  noun .
 3. Max marginal (-142.93): noun verb    adv   conj adv    verb     verb    adp  det  no  un noun .    verb    prt  det  noun .
                          : 0    0       0     0    0      0        0       0    0    0      0    0    0       0    0    0    0
          4. MAP (-276.70): noun verb    adv   conj adv    adv      adv     adv  adv  ad  v  adv  adv  adv     adv  adv  adv  adv
         5. Best (-142.95): noun verb    adv   conj adv    verb     verb    adp  det  no  un noun .    verb    adp  det  noun .

==> So far scored 1 sentences with 17 words.
                   Words correct:     Sentences correct:
   0. Ground truth:      100.00%              100.00%
          1. Naive:       94.12%                0.00%
        2. Sampler:       88.24%                0.00%
   3. Max marginal:       88.24%                0.00%
            4. MAP:       23.53%                0.00%
           5. Best:       82.35%                0.00%
----
                          : desperately ,    nick flashed one  hand up   ,    catching p  oet's neck in   the  bend of   his  elbow .
 0. Ground truth (-136.63): adv         .    noun verb    num  noun prt  .    verb     n  oun   noun adp  det  noun adp  det  noun  .
        1. Naive (-138.34): adv         .    noun verb    num  noun prt  .    verb     n  oun   noun adp  det  verb adp  det  noun  .
      2. Sampler (-136.63): adv         .    noun verb    num  noun prt  .    verb     n  oun   noun adp  det  noun adp  det  noun  .
                 (-136.63): adv         .    noun verb    num  noun prt  .    verb     n  oun   noun adp  det  noun adp  det  noun  .
                 (-136.63): adv         .    noun verb    num  noun prt  .    verb     n  oun   noun adp  det  noun adp  det  noun  .
                 (-136.63): adv         .    noun verb    num  noun prt  .    verb     n  oun   noun adp  det  noun adp  det  noun  .
                 (-136.63): adv         .    noun verb    num  noun prt  .    verb     n  oun   noun adp  det  noun adp  det  noun  .
 3. Max marginal (-136.63): adv         .    noun verb    num  noun prt  .    verb     n  oun   noun adp  det  noun adp  det  noun  .
                          : 0           0    0    0       0    0    0    0    0        0        0    0    0    0    0    0    0     0
          4. MAP (-136.63): adv         .    noun verb    num  noun prt  .    verb     n  oun   noun adp  det  noun adp  det  noun  .
         5. Best (-136.63): adv         .    noun verb    num  noun prt  .    verb     n  oun   noun adp  det  noun adp  det  noun  .

==> So far scored 2 sentences with 35 words.
                   Words correct:     Sentences correct:
   0. Ground truth:      100.00%              100.00%
          1. Naive:       94.29%                0.00%
        2. Sampler:       94.29%               50.00%
   3. Max marginal:       94.29%               50.00%
            4. MAP:       62.86%               50.00%
           5. Best:       91.43%               50.00%
----
                          : the  air  hose was  free !    !
 0. Ground truth ( -50.58): det  noun noun verb adj  .    .
        1. Naive ( -50.58): det  noun noun verb adj  .    .
      2. Sampler ( -52.14): det  noun noun verb adv  .    .
                 ( -52.14): det  noun noun verb adv  .    .
                 ( -52.14): det  noun noun verb adv  .    .
                 ( -50.58): det  noun noun verb adj  .    .
                 ( -50.58): det  noun noun verb adj  .    .
 3. Max marginal ( -50.58): det  noun noun verb adj  .    .
                          : 0    0    0    0    0    0    0
          4. MAP ( -50.58): det  noun noun verb adj  .    .
         5. Best ( -50.58): det  noun noun verb adj  .    .

==> So far scored 3 sentences with 42 words.
                   Words correct:     Sentences correct:
   0. Ground truth:      100.00%              100.00%
          1. Naive:       95.24%               33.33%
        2. Sampler:       92.86%               33.33%
   3. Max marginal:       95.24%               66.67%
            4. MAP:       69.05%               66.67%
           5. Best:       92.86%               66.67%

'''
#=========================================================================================================================
# ***************************************************END OF THE REPORT******************************************************
#_____________________________________________________________________________________________________________________________
####

import random
import math
import operator
import collections

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:

    wordset={}
    tagset={}
    s1_count={}
    prob_s1={}
    si1_si_count={}
    prob_si1_si={}
    w_s_count={}
    prob_w_s={}
    prob_s_w={}
    wordcount=0
    word_tag_prob={}
    sentencecount=0
    
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        posterior=1.0
        result=0.0
        #print "words",self.wordcount
        defaultfactor=1.0/self.wordcount
        for i in range(len(sentence)):
            #print speech_first.get(sentence[i])
            if i ==0:
                if((sentence[i],label[i]) in self.prob_w_s):
                    posterior = math.log(self.prob_s1[label[i]]*self.prob_w_s[(sentence[i],label[i])])
                else:
                    posterior = math.log(self.prob_s1[label[i]]*0.0000000000001)
            else:
                if((sentence[i],label[i]) in self.prob_w_s):
                    #print self.prob_w_s[(sentence[i],label[i])],self.prob_si1_si[(label[i],label[i-1])]
                    posterior = posterior+math.log(self.prob_w_s[(sentence[i],label[i])]*self.prob_si1_si[(label[i],label[i-1])])
                else:
                    posterior = posterior+math.log(defaultfactor*0.006944444)
               
        return posterior

    # Do the training!
    #
    def train(self, data):
        for dataelem in data:
            if(dataelem[1][0] not in self.s1_count):    # dataelem[1][0]:first word tag of every sentence 
                self.s1_count[dataelem[1][0]]=1
            elif(dataelem[1][0] in self.s1_count): 
                self.s1_count[dataelem[1][0]]+=1        
            for word in range(0,len(dataelem[0])):      #take each sentence and count words into wordset
                self.wordcount+=1
                if(dataelem[0][word] not in self.wordset):
                    self.wordset[dataelem[0][word]]=1
                elif(dataelem[0][word] in self.wordset):
                    self.wordset[dataelem[0][word]]+=1
                
                
                if(dataelem[1][word] not in self.tagset):  #take each sentence and count tags into tagset
                    self.tagset[dataelem[1][word]]=1
                elif(dataelem[1][word] in self.tagset):
                    self.tagset[dataelem[1][word]]+=1
                    
                w_s=(dataelem[0][word],dataelem[1][word])    #store all (w,s) tuples
                if(w_s not in self.w_s_count):
                    self.w_s_count[w_s]=1
                elif(w_s in self.w_s_count):
                    self.w_s_count[w_s]+=1
                    
                '''if(dataelem[0][word],dataelem[1][word] not in self.word_tag_count):
                    self.word_tag_count[dataelem[0][word]]={}
                    self.word_tag_count[dataelem[0][word]][dataelem[1][word]]=1
                elif(dataelem[0][word],dataelem[1][word] in self.word_tag_count):
                    self.word_tag_count[dataelem[0][word]][dataelem[1][word]]+=1'''
                
                
                #print "worddict",self.word_tag_count
                if(word>0):
                    si1_si=(dataelem[1][word],dataelem[1][word-1])    #store all (s i+1, s i) tuples
                    if(si1_si not in self.si1_si_count):
                        self.si1_si_count[si1_si]=1
                    if(si1_si in self.si1_si_count):
                        self.si1_si_count[si1_si]+=1
        #print self.tagset            
        for key in self.tagset:  # assign other non occuring si+1,sI pairs count to 1 
            for key1 in self.tagset:
                if((key,key1) not in self.si1_si_count):
                    self.si1_si_count[(key,key1)]=1
                    
        for key in self.tagset:  # assign other non occuring tags as s1 count to 1 
                if key not in self.s1_count:
                    self.s1_count[key]=1
        
        #print self.si1_si_count            
        #print " length sis2", len(self.si1_si_count)
                
        #calculate probability of s1
        for firsttag in self.s1_count:
            self.prob_s1[firsttag]=self.s1_count[firsttag]*(1.0)/len(data)    #p(s1)=count(s1)/total sentences
        
        self.sentencecount=len(data)
        #print len(data)
        #print self.wordcount    
        #calculate probability of si+1/si
        for neigh_tuple in self.si1_si_count:      # neigh_tuple = all (si+1,si) pairs
            self.prob_si1_si[neigh_tuple]=self.si1_si_count[neigh_tuple]*(1.0)/self.tagset[neigh_tuple[1]]    #p(si+1/si)=count(si+1 and si)/count(si)
            
        #calculate probability of w/s
        for ws_tuple in self.w_s_count:      # ws_tuple = all (w,s) pairs
            self.prob_w_s[ws_tuple]=self.w_s_count[ws_tuple]*(1.0)/self.tagset[ws_tuple[1]]    #p(si+1/si)=count(si+1 and si)/count(si)
            
        
        
        #print "s1count",self.s1_count
        #print "words",self.wordset
        #print "probs1",self.prob_s1
        #print "total sentences",len(data)
        #print "tagset",self.tagset
        #print "s1,s1",self.si1_si_count
        #print "probs1,s1",self.prob_si1_si
        #print "prob w,s",self.prob_w_s
        
        
        
        pass

    # Functions for each algorithm.
    #
    def naive(self, sentence):
        tagging=[]
        #print "sentence",sentence
        for wordidx in range(0,len(sentence)):      #take each word and estimate most probable tag
            problist=[]
            keylist=[]
            for tag in self.tagset:
                w_s=(sentence[wordidx],tag)
                if(w_s in self.prob_w_s): 
                    s_w=(tag,sentence[wordidx])
                    probability=self.prob_w_s[w_s]*self.tagset[tag]/self.wordset[sentence[wordidx]]
                    self.prob_s_w[s_w]=probability
                    problist.append(probability)
                    keylist.append(s_w)
            #print problist
            #print keylist
            if(len(problist)==1):
                tag=keylist[0][0]
            elif(len(problist)>1):
                idx=problist.index(max(problist))
                tag=keylist[idx][0]
            tagging.append(tag)
        #print"naive tagging",tagging
        return [ [tagging], [] ]
        #return [ [ [ "noun" ] * len(sentence)], [] ]

    def mcmc(self, sentence, sample_count):
        
        sample=self.naive(sentence)[0][0]
        #print "sample",sample
        sampleset=[]
        sampleset.append(sample)
        taggingset=[]
        smp=sampleset[0]
        #sample=['noun']*len(sentence)
        #print "sample",sample1
        #print sentence,sampleset
        if(len(smp)>1):
            for i in range(0,sample_count):
                tagging=[]
                smp=sampleset[i]
                #calc prob s1
                #print sentence,smp
                w1s1=(sentence[0],smp[0])
                s2s1=(smp[1],smp[0])
                if w1s1 in self.prob_w_s:
                    #print w1,s1
                    #p_s1=self.prob_w_s[w1s1]*self.prob_si1_si[s2s1]*self.tagset[smp[0]]/self.sentencecount
                    p_s1=self.prob_w_s[w1s1]*self.prob_si1_si[s2s1]*self.prob_s1[smp[0]]
                    #print"p_s1",p_s1
                '''prob_list=[]
                key_list=[]
                for tag in self.tagset:
                     wisi=(sentence[0],tag)
                     if wisi in self.prob_w_s:
                         sisi=(smp[1],tag)
                         p_si=self.prob_w_s[wisi]*self.prob_si1_si[sisi]*self.tagset[tag]/self.wordcount
                         prob_list.append(p_si)
                         key_list.append(wisi)
    
                     else:
                         p_si=0.25*0.25*self.tagset[tag]/self.wordcount
                         prob_list.append(p_si)
                         key_list.append(wisi)
                         
                
                #normalisation
                prob_list=self.normalise(prob_list)
                #print "problist",prob_list
                #Cumulative
                cumproblist=[sum(prob_list[:i+1]) for i in xrange(len(prob_list))]
                print "cproblist",cumproblist
                s1=key_list[self.sampling(cumproblist)][1]
                tagging.append(key_list[self.sampling(cumproblist)][1])'''
                s1=smp[0]
                tagging.append(s1)
                if(len(smp)>1):
                    #------------------------------------------------------------------------------------------------
                    #calc prob s2 to sn-1
                    sb=s1
                    for i in range(1,len(sentence)-1):
                      prob_list=[]
                      key_list=[]
                      for tag in self.tagset:
                          wisi=(sentence[i],tag)
                          sisb=(tag,sb)
                          sasi=(smp[i+1],tag)
                          if(wisi in self.prob_w_s):
                              #print wisi,"present in"
                              p_si_wisbsa=self.prob_w_s[wisi]*self.prob_si1_si[sisb]*self.prob_si1_si[sasi]
                              prob_list.append(p_si_wisbsa)
                              key_list.append(wisi)
                          else:
                              #print wisi,"not in"
                              p_si_wisbsa=0.0*self.prob_si1_si[sisb]*self.prob_si1_si[sasi]
                              prob_list.append(p_si_wisbsa)
                              key_list.append(wisi)
                      
                      #print " problist", prob_list
                      #print " keylist", key_list
                      #normalisation
                      prob_list=self.normalise(prob_list)
        
                      #Cumulative
                      cumproblist=[sum(prob_list[:i+1]) for i in xrange(len(prob_list))]
                      #print "cproblist",cumproblist
                      sb=key_list[self.sampling(cumproblist)][1]
                      tagging.append(key_list[self.sampling(cumproblist)][1])
                    
                    #------------------------------------------------------------------------------------------------
                    #calc prob sn
                    sn_1=sb
                    prob_list=[]
                    key_list=[]
                    for tag in self.tagset:
                        n=len(sentence)-1
                        wnsn=(sentence[n],tag)
                        snsn_1=(tag,sn_1) 
                        if wnsn in self.prob_w_s:
                            
                            p_sn=self.prob_w_s[wnsn]*self.prob_si1_si[snsn_1]
                            prob_list.append(p_sn)
                            key_list.append(wnsn)
                        else:
                             p_sn=0.0*self.prob_si1_si[snsn_1]
                             prob_list.append(p_sn)
                             key_list.append(wnsn)
                    #normalisation
                    prob_list=self.normalise(prob_list)
                    #print "problist",prob_list
                    #Cumulative
                    cumproblist=[sum(prob_list[:i+1]) for i in xrange(len(prob_list))]
                    #print "cproblist",cumproblist
                    tagging.append(key_list[self.sampling(cumproblist)][1])
                sampleset.append(tagging)
                taggingset.append(tagging)
            #print "tagging",tagging                   
            return [ taggingset, [] ]
        else:
            return self.naive(sentence)

    def best(self, sentence):
        a= self.mcmc(sentence, 10)
        lastindex=len(a[0])-1
        tagging = a[0][lastindex]
        return [ [tagging], [] ]

    def max_marginal(self, sentence):
        sample=self.viterbi(sentence)[0][0]
        #print "sample",sample
        sampleset=[]
        sampleset.append(sample)
        taggingset=[]
        wordaccuracy=[]
        smp=sampleset[0]
        #print sentence,sampleset
        if(len(smp)>1):
            for i in range(0,10):
                #print "-------", i
                tagging=[]
                smp=sampleset[i]
                #calc prob s1
                #print sentence,smp
                w1s1=(sentence[0],smp[0])
                s2s1=(smp[1],smp[0])
                if w1s1 in self.prob_w_s:
                    #print w1,s1
                    p_s1=self.prob_w_s[w1s1]*self.prob_si1_si[s2s1]*self.prob_s1[smp[0]]
                    if(i==999):
                        wordaccuracy.append(p_s1)
                    #print"p_s1",p_s1
                '''prob_list=[]
                key_list=[]
                for tag in self.tagset:
                     wisi=(sentence[0],tag)
                     if wisi in self.prob_w_s:
                         sisi=(smp[1],tag)
                         p_si=self.prob_w_s[wisi]*self.prob_si1_si[sisi]*self.tagset[tag]/self.wordcount
                         prob_list.append(p_si)
                         key_list.append(wisi)
    
                     else:
                         p_si=0.25*0.25*self.tagset[tag]/self.wordcount
                         prob_list.append(p_si)
                         key_list.append(wisi)
                         
                
                #normalisation
                prob_list=self.normalise(prob_list)
                #print "problist",prob_list
                #Cumulative
                cumproblist=[sum(prob_list[:i+1]) for i in xrange(len(prob_list))]
                print "cproblist",cumproblist
                s1=key_list[self.sampling(cumproblist)][1]
                tagging.append(key_list[self.sampling(cumproblist)][1])'''
                s1=smp[0]
                tagging.append(s1)
                if(len(smp)>1):
                    #------------------------------------------------------------------------------------------------
                    #calc prob s2 to sn-1
                    sb=s1
                    for i in range(1,len(sentence)-1):
                      prob_list=[]
                      key_list=[]
                      for tag in self.tagset:
                          wisi=(sentence[i],tag)
                          sisb=(tag,sb)
                          sasi=(smp[i+1],tag)
                          if(wisi in self.prob_w_s):
                              p_si_wisbsa=self.prob_w_s[wisi]*self.prob_si1_si[sisb]*self.prob_si1_si[sasi]
                              prob_list.append(p_si_wisbsa)
                              key_list.append(wisi)
                          else:
                              p_si_wisbsa=0.0*self.prob_si1_si[sisb]*self.prob_si1_si[sasi]
                              prob_list.append(p_si_wisbsa)
                              key_list.append(wisi)
                      #normalisation
                      prob_list=self.normalise(prob_list)
        
                      #Cumulative
                      cumproblist=[sum(prob_list[:i+1]) for i in xrange(len(prob_list))]
                      #print "cproblist",cumproblist
                      sb=key_list[self.sampling(cumproblist)][1]
                      tagging.append(key_list[self.sampling(cumproblist)][1])
                      if(i==999):
                          wordaccuracy.append(prob_list[sb])
                    
                    #------------------------------------------------------------------------------------------------
                    #calc prob sn
                    sn_1=sb
                    prob_list=[]
                    key_list=[]
                    for tag in self.tagset:
                        n=len(sentence)-1
                        wnsn=(sentence[n],tag)
                        snsn_1=(tag,sn_1) 
                        if wnsn in self.prob_w_s:
                            
                            p_sn=self.prob_w_s[wnsn]*self.prob_si1_si[snsn_1]
                            if(i==999):
                                wordaccuracy.append(prob_list[p_sn])
                            prob_list.append(p_sn)
                            key_list.append(wnsn)
                        else:
                             p_sn=0.0*self.prob_si1_si[snsn_1]
                             if(i==999):
                                 wordaccuracy.append(prob_list[p_sn])
                             prob_list.append(p_sn)
                             key_list.append(wnsn)
                    #normalisation
                    prob_list=self.normalise(prob_list)
                    #print "problist",prob_list
                    #Cumulative
                    cumproblist=[sum(prob_list[:i+1]) for i in xrange(len(prob_list))]
                    #print "cproblist",cumproblist
                    tagging.append(key_list[self.sampling(cumproblist)][1])
                    sampleset.append(tagging)    
            #print "tagging",tagging
            #print "word accuracy", wordaccuracy                   
            return [ [tagging], [[0] * len(sentence),] ]
            #return [ [ [ "noun" ] * len(sentence)], [[0] * len(sentence),] ]
        else:
            return self.naive(sentence)

    def viterbi(self, sentence):
        #print self.prob_w_s
        word=sentence[0]
        problist=[]
        keylist=[]
        maxprob=1
        prevtag='noun'
        tagging=[]
        for tag in self.tagset:
            #print (word,tag)
            if((word,tag) in self.prob_w_s):
                #print "present in "
                p_s1w = self.prob_s1[tag]*self.prob_w_s[(word,tag)]*(1.0) #/self.sentencecount
            else:
                #print"not present in probws"
                p_s1w = 0.0 #(self.s1_count[tag]/self.wordcount)*0.0    
            problist.append(p_s1w)
            keylist.append((word,tag))
        #print "key",keylist
        #print "prob",problist
        maxprob=max(problist)
        i=problist.index(maxprob)
        #print "maxid",i
        tagging.append(keylist[i][1])
        prevtag=keylist[i][1]
        if(len(sentence)>1):
            for wordidx in range(1,len(sentence)):
                word=sentence[wordidx]
                #print "prev tag",prevtag
                problist=[]
                keylist=[]
                for tag in self.tagset:
                    #print(word,tag)
                    #print(tag,prevtag)
                    if((word,tag) in self.prob_w_s):
                        p_siwi = maxprob*self.prob_w_s[(word,tag)]*self.prob_si1_si[(tag,prevtag)]
                    else:
                        p_siwi = 0.0#maxprob*0.00083*self.prob_si1_si[(tag,prevtag)]
                    problist.append(p_siwi)
                    keylist.append((word,tag))
                #print keylist
                #print problist
                maxprob=max(problist)
                i=problist.index(maxprob)
                #print "maxid",i
                tagging.append(keylist[i][1])
                prevtag=keylist[i][1]
                
        #print "tagging viterbi", tagging
            
        return [ [tagging], [] ]    
        #return [ [ [ "noun" ] * len(sentence)], [] ]


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It's supposed to return a list with two elements:
    #
    #  - The first element is a list of part-of-speech labelings of the sentence.
    #    Each of these is a list, one part of speech per word of the sentence.
    #    Most algorithms only return a single labeling per sentence, except for the
    #    mcmc sampler which is supposed to return 5.
    #
    #  - The second element is a list of probabilities, one per word. This is
    #    only needed for max_marginal() and is the marginal probabilities for each word.
    #
    def solve(self, algo, sentence):
        if algo == "Naive":
            return self.naive(sentence)
        elif algo == "Sampler":
            return self.mcmc(sentence, 5)
        elif algo == "Max marginal":
            return self.max_marginal(sentence)
        elif algo == "MAP":
            return self.viterbi(sentence)
        elif algo == "Best":
            return self.best(sentence)
        else:
            print "Unknown algo!"
            
    def sampling(self,list1):
        index=0
        for n in range(0, 1000):
           x=random.random()
           for i in range(len(list1)):
              if  x <= list1[i]:
                  index=i
                  break;
        return index
    def normalise(self,prob_list):
        totalp=sum(prob_list)
        if (totalp>0):
            for i in range(len(prob_list)):
                prob_list[i]=prob_list[i]*(1.0)/totalp
        return prob_list
        
    '''def defaulttagger(self,prev_tag):
        problist=[]
        keylist=[]
        for tag in self.tagset:
            key=(tag,prev_tag)
            problist.append(self.prob_si1_si[key])
            keylist.append(key)
        id = problist.index(max(problist))
        return keylist[id]'''