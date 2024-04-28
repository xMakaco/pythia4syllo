<p style="text-align: center;">
                                   _ _                       _             
           .___,         ___ _   _| | | ___  _ __ ___   __ _| | _____ _ __         .___,
        ___('v')___     / __| | | | | |/ _ \| '_ ` _ \ / _` | |/ / _ \ '__|     ___('v')___    
        `"-\._./-"´     \__ \ |_| | | | (_) | | | | | | (_| |   <  __/ |        `"-\._./-"´
            ^ ^         |___/\__, |_|_|\___/|_| |_| |_|\__,_|_|\_\___|_|            ^ ^ 
            '''              |___/                                                  '''
</p>
                                                                  
# Syllomaker
A program made to generate sets of syllogisms given the mood and figure of the premises.
Especially useful if you need to create datasets for syllogistic reasoning tasks. 😉

## Arguments
Syllomaker can take different arguments when called from the terminal.
```
python3 syllomaker.py [-m|--mood] XY [-f|--figure] Z ([-n|--num] N | [-i|--items] M,P,S) ([-d|--datafile] FILEPATH.TSV)
```
where
- [-m | --mood] XY is the mood of the two premises, composed by two letters in {A, I, E, O}.
- [-f | --figure] Z is the figure of the premises, which needs to be one (or more) number(s) in [1, 4] or 'r'.
    - One number in [1, 4]: the figure of all syllogisms will be the specified number.
    - More than one number in [1, 4]: the figures of the syllogisms will be randomised between the numbers selected.
    - Random option 'r': the figures of the syllogisms will be randomised, so `-f r` and `-f 1234` are equivalent.
- [-n | --num] N is the number of syllogisms we want to generate given a mood and figure. It defaults to `N = 5` and can only be used in automatic mode, i.e. when --items are not given.
- [-i | --items] M,P,S are the middle, predicate and subject of the syllogism. When items are not given by the user, the program takes words from WordNet.
- [-d | --datafile] FILEPATH.TSV is the path of the file where we want to store the syllogisms. It defaults to syllogism_data.tsv.

## Some examples  
By default, syllomaker.py automatically creates nonsense syllogisms with words taken from WordNet, so only the mood and figure are mandatory arguments.
```
python3 syllomaker.py --mood ([A|I|E|O]*2) --figure ([1|2|3|4]+|r) 
```
For example, if we want 10 syllogisms of the form IA3, we would input
```
python3 syllomaker.py --mood AI --figure 3 --num 10
```
However, it can also take as an argument a list of items separated by commas to create **one** syllogism. 
```
python3 syllomaker.py --mood AI --figure 3 --items m,p,s
```
Note that the items are in a specific order - middle, predicate, subject - separated by commas without spaces, e.g.
```
python3 syllomaker.py --mood AI --figure 3 --items philosophers,great,greeks
```

## The output
Given the mood and figure of the two premises, Syllomaker generates the valid conclusion(s) or "NVC" (No Valid Conclusion), and stores all this data in a .tsv file. The dataset generated with Syllomaker will look like this:
| Type | Premise 1 | Premise 2 | Conclusion | Conclusion 2 |
| :----| :-------- | :-------- | :--------- | :----------- |
| AI3  | All philosophers are great. | Some philosophers are Greeks. | Some Greeks are great. |  |
| AA1  | All cats are pets. | All pets are beautiful. | All cats are beautiful. | Some cats are beautiful. |
| EO4  | No tiramisùs are pizzas. | Some pizzas are not cheap. | NVC |  |
| ...  | ...       | ...       | ...        | ...          |

## Dependencies
The only libraries needed are nltk and docopt. To install them, run either
```
pip install nltk docopt
```
or simply
```
pip install -r requirements.txt
```
