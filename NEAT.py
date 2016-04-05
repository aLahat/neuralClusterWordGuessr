# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 17:17:00 2016

@author: Rachel
"""

""" 2-input XOR example """
from __future__ import print_function
import string
import random
from neat import nn, population, statistics, visualize

# Network inputs and expected outputs.
xor_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
xor_outputs = [0, 1, 1, 0]

outputLength = 5
tests=50
letters = list(string.lowercase)
letters = map(lambda x: ord(x)-97,letters)
print (letters)
words = open('words.txt').read().split('\n')
words = filter(lambda x: len(x)==outputLength ,words)
print((len(letters),len(words)))


#%%
def decrypt(n):
    out = ''
    for i in n:
        i = int((i)*len(letters))%len(letters)+ord('a')
        out+=chr(i)
    return out
def testGenome(genome, v=False,summary=False):
    g = genome    
    net = nn.create_feed_forward_phenotype(g)
    wordsFound=[]
    allwords=[]
    score = 0.0
    for i in range(tests):
        letters = [random.randint(0,1) for i in range(26)]
        output = net.serial_activate(letters)
        word = decrypt(output)
        allwords.append(word)
        if word in words:
            wordsFound.append(word)
            
    wordsFound = set(wordsFound)
    if v: print(wordsFound)
    profile = ([len(set(map(lambda x: x[i],wordsFound))) for i in range(outputLength)])
    profile = sum([x/float(len(letters)) for x in profile])/(outputLength)

    score  = 0.5*profile
    score += 0.5*len(wordsFound)/float(tests)
    if summary:
        for i in allwords:
            if i in words:print('%s - GOOD'%i)
            else:print('%s - BAD'%i)
        print (score)   
    return score
def eval_fitness_words(genomes):
    for g in genomes:
        g.fitness = testGenome(g)

#%%

pop = population.Population('xor2_config')
backup=pop
#%%


while True:
    pop.run(eval_fitness_words, 1)
    backup=pop
    #print('Number of evaluations: {0}'.format(pop.total_evaluations))
    
    # Display the most fit genome.
    winner = pop.statistics.best_genome()
    #print('\nBest genome:\n{!s}'.format(winner))
    testGenome(winner,v=True,summary=False)
#%%
#pop = backup
# Verify network output against training data.
print('\nOutput:')
winner = pop.statistics.best_genome()
testGenome(winner,summary=True)

#%%


winner_net = nn.create_feed_forward_phenotype(winner)
for i in range(tests):
    random.shuffle(letters)
    output = (winner_net.serial_activate(letters))
    #print(output)
    output = decrypt(output)
    if output in words:print('%s- GOOD'%output)
    else:print('%s- BAD'%output)

# Visualize the winner network and plot/log statistics.
visualize.plot_stats(pop.statistics)
visualize.plot_species(pop.statistics)
visualize.draw_net(winner, view=True, filename="xor2-all.gv")
visualize.draw_net(winner, view=True, filename="xor2-enabled.gv", show_disabled=False)
visualize.draw_net(winner, view=True, filename="xor2-enabled-pruned.gv", show_disabled=False, prune_unused=True)
statistics.save_stats(pop.statistics)
statistics.save_species_count(pop.statistics)
statistics.save_species_fitness(pop.statistics)
#%%