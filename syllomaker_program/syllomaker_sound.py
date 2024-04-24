import utils

# GET THE NOUNS FOR THE SYLLOGISM FROM THE USER
items = input("give three items (m,p,s) separated by commas :)")
itemlist = list(items.split(','))

if len(itemlist) != 3:
    raise TypeError("i said three things!!! >:(")

x = itemlist[0]
y = itemlist[1]
z = itemlist[2]

# GET THE FIGURE FOR THE SYLLOGISM FROM THE USER
figure = int(input("give me a figure :)"))
print(type(figure))

if figure not in [1,2,3,4]:
    raise TypeError("that is not a valid figure :(")

# ACTUALLY CREATE THE SYLLOGISM
dataline = make_syllogism(x, y, z, figure)
print(dataline)

# ADD IT TO THE TSV FILE
file = open('yourfilepath.tsv ', 'a')
file.write(dataline)
file.write("\n")
