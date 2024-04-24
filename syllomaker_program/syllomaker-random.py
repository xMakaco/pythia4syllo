import random
from utils import *

# LOOP FOR THE NUMBER OF SYLLOGISMS WE WANT
n = 0
while n < 100:

    # ASSIGN A RANDOM WORD FROM WORDNET TO THREE VARIABLES
    x = make_word()
    y = make_word()
    z = make_word()
    #print(x); print(y); print(z)

    # GET A RANDOM FIGURE
    fig = random.choice([2,3,4])

    # CREATE A SYLLOGISM
    dataline = make_barbara(x, y, z, fig)
    print(dataline)

    # ADD IT TO THE TSV FILE
    file = open('syllogism_data.tsv', 'a')
    file.write(dataline)
    file.write("\n")

    # ADD 1 TO THE LOOP VARIABLE
    n +=1
