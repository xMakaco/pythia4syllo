from utils import *

# GET THE NOUNS FOR THE SYLLOGISM
items = input("give three items (med,pred,sub) separated by commas :) ")
itemlist = list(items.split(','))

if len(itemlist) != 3:
    raise TypeError("i said three items!!! >:( ")

x = itemlist[0]
y = itemlist[1]
z = itemlist[2]

# GET THE FIGURE FOR THE SYLLOGISM
figure = int(input("give me a figure :) "))

if figure not in [1,2,3,4]:
    raise TypeError("that is not a valid figure :( ")

# ACTUALLY CREATE THE SYLLOGISM
dataline = make_barbara(x, y, z, figure)
print(dataline)

# ADD IT TO THE TSV FILE
file = open('syllogism_data.tsv', 'a')
file.write(dataline)
file.write("\n")
