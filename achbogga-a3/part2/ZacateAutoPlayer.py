# Automatic Zacate game player
# B551 Fall 2015
# PUT YOUR NAME AND USER ID HERE!
# Achyut Sarma Boggaram | achbogga
#
# Based on skeleton code by D. Crandall
#
# PUT YOUR REPORT HERE!
# I have written two functions to give good reroll and good category based on the expected score. I have tried to make a good reroll by the help of expected scores of each category. 
# 
#
# This is the file you should modify to create your new smart Zacate player.
# The main program calls this program three times for each turn. 
#   1. First it calls first_roll, passing in a Dice object which records the
#      result of the first roll (state of 5 dice) and current Scorecard.
#      You should implement this method so that it returns a (0-based) list 
#      of dice indices that should be re-rolled.
#   
#   2. It then re-rolls the specified dice, and calls second_roll, with
#      the new state of the dice and scorecard. This method should also return
#      a list of dice indices that should be re-rolled.
#
#   3. Finally it calls third_roll, with the final state of the dice.
#      This function should return the name of a scorecard category that 
#      this roll should be recorded under. The names of the scorecard entries
#      are given in Scorecard.Categories.
#

from ZacateState import Dice
from ZacateState import Scorecard
import random
from copy import deepcopy

class ZacateAutoPlayer:

      def __init__(self):
            pass  
      '''def check_repeats(self, dice, scorecard):
			unFilledCategories=deepcopy(list(set(Scorecard.Categories) - set(scorecard.scorecard.keys())))
			print "\n"+str(unFilledCategories)'''
      
      
      def give_me_current_good_cat(self, dice, scorecard):
		  cat_wise_scores=[]
		  unFilledCategories=deepcopy(list(set(Scorecard.Categories) - set(scorecard.scorecard.keys())))
		  for i in range(len(unFilledCategories)):
			  c=scorecard.category_wise_exp_if_bonus_and_score(unFilledCategories[i],dice)
			  cat_wise_scores.append(c)
		  return (unFilledCategories[cat_wise_scores.index(max(cat_wise_scores))])
		  
      def give_me_current_good_reroll(self, dice, scorecard):
      # define the function blocks
		# Categories = [ "unos", "doses", "treses", "cuatros", "cincos", "seises", "pupusa de queso", "pupusa de frijol", "elote", "triple", "cuadruple", "quintupulo", "tamal" ]
		unFilledCategories=deepcopy(list(set(Scorecard.Categories) - set(scorecard.scorecard.keys())))
		category=self.give_me_current_good_cat(dice, scorecard);
		'''for i in range(len(unFilledCategories)):
			if(unFilledCategories[i]=='')
		def unos():
		    #print "Unos_favoured_reroll\n"
		
		def doses():
		    #print "doses_favoured_reroll\n"
		
		def treses():
		    #print "treses_favoured_reroll\n"
		
		def cuatros():
		    #print "cuatros_favoured_reroll\n"
		
		def cincos():
		    #print "cincos_favoured_reroll\n"
		
		def seises():
		    #print "seises_favoured_reroll\n"
		
		def pupusa_de_queso():
		    #print "pupusa_de_queso_favoured_reroll\n"
		
		def pupusa_de_frijol():
		    #print "pupusa de frijol_favoured_reroll\n"
		
		def elote():
		    #print "elote_favoured_reroll\n"
		
		def triple():
		    #print "triple_favoured_reroll\n"
		
		def cuadruple():
		    #print "cuadruple_favoured_reroll\n"
		
		def quintupulo():
		    #print "quintupulo_favoured_reroll\n"
		
		def tamal():
		    #print "tamal_favoured_reroll\n"
		
		# map the inputs to the function blocks
		options = {0 : unos,
		           1 : doses,
		           2 : treses,
		           3 : cuatros,
		           4 : cincos,
		           5 : seises,
		           6 : pupusa_de_queso,
		           8 : pupusa_de_frijol,
		           9 : elote,
		           10 : triple,
		           11 : cuadruple,
		           12 : quintupulo,
		           13 : tamal
		}
		#call-> options[num]()
		'''
		if category in scorecard.scorecard:
			print "Error: category already full!"
		reroll=[]
		if category in scorecard.Numbers:
			if (category == "unos"):
				d=deepcopy(dice.dice)
				d = [x for x in d if x != 1]#reroll all 1's as the score is very minimum
				for i in range(len(d)):
					reroll.append(dice.dice.index(d[i]))
			elif (category == "doses"):
				d=deepcopy(dice.dice)
				d = [x for x in d if x != 2] 
				for i in range(len(d)):
					reroll.append(dice.dice.index(d[i]))
			elif (category == "treses"):
				d=deepcopy(dice.dice)
				d = [x for x in d if x != 3]
				for i in range(len(d)):
					reroll.append(dice.dice.index(d[i]))
			elif (category == "cuatros"):
				d=deepcopy(dice.dice)
				d = [x for x in d if x != 4]
				for i in range(len(d)):
					reroll.append(dice.dice.index(d[i]))
			elif (category == "cincos"):
				d=deepcopy(dice.dice)
				d = [x for x in d if x != 5]
				for i in range(len(d)):
					reroll.append(dice.dice.index(d[i]))
			elif (category == "seises"):
				d=deepcopy(dice.dice)
				d = [x for x in d if x != 6]
				for i in range(len(d)):
					reroll.append(dice.dice.index(d[i]))
			return reroll
		elif category == "pupusa de queso":
			return []
		elif category == "pupusa de frijol":
			'''if (len(set([1,2,3,4]) - set(dice.dice)) == 0):
				return [dice.dice.index(list(set(dice.dice)-set([1,2,3,4])))]#not working as anticipated
			elif (len(set([2,3,4,5]) - set(dice.dice)) == 0):
				return [dice.dice.index(list(set(dice.dice)-set([2,3,4,5])))]
			elif (len(set([3,4,5,6]) - set(dice.dice)) == 0):
				return [dice.dice.index(list(set(dice.dice)-set([3,4,5,6])))]'''
			return reroll
		elif category == "elote":
			return reroll
		elif category == "triple":
			c=[0,0,0,0,0]
			for i in range(len(dice.dice)):
				for j in range(len(dice.dice)):
					if (dice.dice[i]==dice.dice[j]):
						c[i] += 1
			for i in range(len(dice.dice)):
				if (c[i]!=(3)):
					reroll.append(i)
			return reroll
		elif category == "cuadruple":
			c=[0,0,0,0,0]
			for i in range(len(dice.dice)):
				for j in range(len(dice.dice)):
					if (dice.dice[i]==dice.dice[j]):
						c[i] += 1
			for i in range(len(dice.dice)):
				if (c[i]!=(4)):
					reroll.append(i)
			return reroll
		elif category == "quintupulo":
			return reroll
		elif category == "tamal":
			d=sorted(dice.dice)
			reroll.append(dice.dice.index(d[0]))
			reroll.append(dice.dice.index(d[1]))
			reroll=sorted(reroll)
			return reroll
		else:
			print "Error: unknown category"
      
      def first_roll(self, dice, scorecard):
		#first reroll algorithm
		'''
		Check all the unfilled categories and select the highest ranked one for the current config.
		'''
		
			
      
		#return [0] # always re-roll first die (blindly)
		return self.give_me_current_good_reroll(dice, scorecard)

      def second_roll(self, dice, scorecard):
            #return [1, 2] # always re-roll second and third dice (blindly)
            return self.give_me_current_good_reroll(dice, scorecard)
      
      def third_roll(self, dice, scorecard):
            # stupidly just randomly choose a category to put this in
            
			
            #return random.choice( list(set(Scorecard.Categories) - set(scorecard.scorecard.keys())) )
            return (self.give_me_current_good_cat(dice, scorecard))

