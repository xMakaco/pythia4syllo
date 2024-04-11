from nltk.corpus import wordnet as wn
import random
import utils

#GET THE FIGURE FROM THE USER  
figure = int(input("give me a figure :)"))
print(type(figure))

if figure not in [1,2,3,4]:
    raise TypeError("that is not a valid figure :(")

#FROM NLTK, DOWNLOAD THE WORDNET CORPUS
nltk.download('wordnet')

#GET A LIST OF ALL THE SYNSETS IN WORDNET
synsets = list(wn.all_synsets())

#ASSIGN A RANDOM WORD FROM WORDNET TO THREE VARIABLES
x = random.choice(synsets).name().split('.')[0].replace('_', ' ')
y = random.choice(synsets).name().split('.')[0].replace('_', ' ')
z = random.choice(synsets).name().split('.')[0].replace('_', ' ')

#ACTUALLY CREATE THE SYLLOGISM
dataline = make_syllogism(x, y, z, figure)
    print(dataline)

# ADD IT TO THE TSV FILE
file = open('yourfilepath.tsv ', 'a')
file.write(dataline)
file.write("\n")
