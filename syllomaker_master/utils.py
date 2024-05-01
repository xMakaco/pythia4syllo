from nltk.corpus import wordnet as wn # type: ignore
import random
import sys
import nltk

try:
    nltk.data.find("corpora/wordnet.zip")
except:
    nltk.download('wordnet') # needed once only

##### CHECK ARGUMENTS GIVEN #####

def check_args(mood, n, filename):
    # CHECK MOOD
    if (mood[0] or mood[1]) not in ['A', 'E', 'I', 'O']:
        raise TypeError("Mood not valid! 😓")
    
    # CHECK NUMBER OF SYLLOGISMS
    if n < 1:
        print("So you do not want syllogisms? ☹️")
    elif n > 1000:
        print("Be careful, that's a lot! 😮")
        surelol = input(f"Are you sure you want {n} syllogisms? yes / no ")
        if surelol == ('y' or 'yes'):
            pass
        else: 
            sys.exit()

    # CHECK FOR EMPTY FILE
    fileread = open(filename, 'r+')

    # ADD COLUMN HEADERS 
    if fileread.read(1) == '':
        fileread.write(f"type\tpremise_1\tpremise_2\tconclusion\tconclusion_2\n")

    return 


##### AUTO MODE VOCABULARY #####

# GET A LIST OF ALL SYNSETS IN WORDNET WITH NOUNS
synsets = list(wn.all_synsets('n'))

def make_word():
    # GET A RANDOM WORD FROM THOSE SYNSETS
    word = random.choice(synsets).name().split('.')[0].replace('_', ' ')
    # QUICKLY AND NON-EXHAUSTIVELY PLURALISE WORDS
    word = word + 'es' if word[-1] in ['s', 'x', 'z', 'y'] or word[-2:] in ['sh', 'ch'] else word + 's'
    
    return word


##### SYLLOGISMS #####

quantifiers = {'A':'All', 'E':'No', 'I':'Some', 'O':'Some'}

def make_premises(mood, fig, m, p, s):
    # TAKE THE MOOD OF PREMISES AND TURN IT INTO QUANTIFIERS
    quant1 = quantifiers.get(mood[0])
    quant2 = quantifiers.get(mood[1])
    polarity1 = 'not ' if mood[0] == 'O' else ''
    polarity2 = 'not ' if mood[1] == 'O' else ''

    # CREATE PREMISES ACORDING TO FIGURE
    if fig == 1:
        prem1 = f"{quant1} {m} are {polarity1}{p}."; prem2 = f"{quant2} {s} are {polarity2}{m}."
    elif fig == 2:
        prem1 = f"{quant1} {p} are {polarity1}{m}."; prem2 = f"{quant2} {s} are {polarity2}{m}."
    elif fig == 3:
        prem1 = f"{quant1} {m} are {polarity1}{p}."; prem2 = f"{quant2} {m} are {polarity2}{s}."
    elif fig == 4:
        prem1 = f"{quant1} {p} are {polarity1}{m}."; prem2 = f"{quant2} {m} are {polarity2}{s}."
    
    return prem1, prem2


def make_conclusion(mood, fig, p, s):
    # GET THE VALID CONCLUSION
    premtype = f'{mood}{fig}'
    
    valid_lists = [
        ['A', 'AA1'], 
        ['I', 'AA1', 'AA3', 'AA4', 'AI1', 'AI3', 'IA3', 'IA4'], 
        ['E', 'AE4', 'EA1', 'EA2'], 
        ['O', 'AE2', 'AE4', 'EA1', 'EA2', 'EA3', 'EA4', 'AO2', 'OA3', 'EI1', 'EI2', 'EI3', 'EI4']
        ] # these are all the valid syllogisms classified by the mood of their valid conclusion

    for list in valid_lists:
        if premtype in list:
            quant_conc = quantifiers.get(list[0])
            polar_conc = 'not ' if list[0] == 'O' else ''
            conc = f"{quant_conc} {s} are {polar_conc}{p}."
            break   # it is necessary to break the for loop once the type is found
        conc = "NVC"

    # ACCOUNT FOR THOSE WITH MORE THAN ONE VALID CONCLUSION
    double_val = ['AA1', 'AE4', 'EA1', 'EA2']
    if premtype in double_val:
        quant_conc2 = 'Some'
        polar_conc2 = '' if premtype == 'AA1' else 'not '
        conc2 = f"{quant_conc2} {s} are {polar_conc2}{p}."
    else: conc2 = ""

    return conc, conc2


def make_syllogism(mood, fig, m, p, s):
    # GET FIGURE
    if fig == 'r': fig = random.choice([1, 2, 3, 4]) 
    else:
        for f in fig:
            if f not in ['1', '2', '3', '4']:
                raise TypeError("Figure not valid! 😓")
        fig = int(fig) if len(fig) == 1 else random.choice([int(f) for f in fig])
        
    premise1, premise2 = make_premises(mood, fig, m, p, s)
    conclusion, conclusion2 = make_conclusion(mood, fig, p, s)
   
    syllogism = f"{mood}{fig}\t{premise1}\t{premise2}\t{conclusion}\t{conclusion2}"

    return syllogism
