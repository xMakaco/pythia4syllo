"""
Syllomaker has been built to create sets of nonsense valid syllogisms.
These sets can be used for probing the abstract reasoning of humans or language models.
Read README.md for more information.

Usage:
    syllomaker.py [options]
    syllomaker.py (-h | --help)

Options:
    -h --help                               Show this screen.
    -v, --version                           Show version.
    -m, --mood ([A|I|E|O]*2)                Mood of the premises, two letters in {A, I, E, O}, e.g. AA, AO, EI, etc.
    -f, --figure ([1|2|3|4]+|r)             Figure of the premises. Can be one of more numbers in [1, 4] or 'r' for random.
    -n, --num <int>                         Amount of syllogisms to create for the given mood (only in auto mode). [default: 3]
    -i, --items <m,p,s>                     List of items to use in one syllogism, separated by commas without spaces. [default: ]
    -d, --datafile <filename>               Name of the file where the syllogisms will be stored. [default: syllogism_data.tsv]
"""

from docopt import docopt
from utils import *

args = docopt(__doc__, version = 'Syllomaker 4.1 - Now with more syllogisms! 😉')
mood = args['--mood']
fig = args['--figure']
n = int(args['--num'])
items = args['--items']
filename = args['--datafile']
file = open(filename, 'a+')

check_args(mood, n, filename)

# IF THERE ARE NO ITEMS GIVEN, WE GO INTO THE AUTOMATIC LOOP
if items == '':
    # LOOP FOR THE NUMBER OF SYLLOGISMS WE WANT FOR THIS MOOD
    i = 0
    while i < n:
        # THINGS HERE WILL BE DONE FOR EACH SYLLOGISM OF THE BUNCH
        # ASSIGN A RANDOM WORD FROM WORDNET TO THREE VARIABLES
        x = make_word()
        y = make_word()
        z = make_word()

        # CREATE A SYLLOGISM AND ADD IT TO THE TSV FILE
        dataline = make_syllogism(mood, fig, x, y, z)
        print(dataline)
        file.write(f"{dataline}\n")

        # ADD 1 TO THE LOOP VARIABLE
        i += 1
    
# IF THERE ARE ITEMS GIVEN AS ARGUMENTS, THESE WILL FORM THE SYLLOGISM
else:
    itemlist = list(items.split(','))
    if len(itemlist) != 3:
        raise TypeError("Manual input should be three items separated by commas! ☹️")
    x = itemlist[0]
    y = itemlist[1]
    z = itemlist[2]
    
    dataline = make_syllogism(mood, fig, x, y, z)
    print(dataline)
    file.write(f"{dataline}\n")
    