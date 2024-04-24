#FUNCTION TO GIVE A SYLLOGISM THE AAA-1 STRUCTURE AND THE VALID LABLE
def make_barbara1(m, p, s):
    prop1 = f"all {m} are {p}"
    prop2 = f"all {s} are {m}"
    conc = f"all {s} are {p}"
    syl = f"{prop1}\t{prop2}\t{conc}\tVALID"
    return syl
  
#FUNCTION TO GIVE A SYLLOGISM THE AAA-2 STRUCTURE AND THE INVALID LABLE
def make_barbara2(m, p, s):
    prop1 = f"all {p} are {m}"
    prop2 = f"all {s} are {m}"
    conc = f"all {s} are {p}"
    syl = f"{prop1}\t{prop2}\t{conc}\tINVALID"
    return syl

#FUNCTION TO GIVE A SYLLOGISM THE AAA-3 STRUCTURE AND THE INVALID LABLE
def make_barbara3(m, p, s):
    prop1 = f"all {m} are {p}"
    prop2 = f"all {m} are {s}"
    conc = f"all {s} are {p}"
    syl = f"{prop1}\t{prop2}\t{conc}\tINVALID"
    return syl

#FUNCTION TO GIVE A SYLLOGISM THE AAA-4 STRUCTURE AND THE INVALID LABLE
def make_barbara4(m, p, s):
    prop1 = f"all {p} are {m}"
    prop2 = f"all {m} are {s}"
    conc = f"all {s} are {p}"
    syl = f"{prop1}\t{prop2}\t{conc}\tINVALID"
    return syl

#FUNCTION TO RETURN A SYLLOGISM, TAKES MIDDLE TERM, MAJOR TERM, MINOR TERM AND FIGURE AS ARGUMENTS
def make_syllogism(m, p, s, fig):
    if fig == 1:
        output = make_barbara1(m, p, s)
    if fig == 2:
        output = make_barbara2(m, p, s)
    if fig == 3:
        output = make_barbara3(m, p, s)
    if fig == 4:
        output = make_barbara4(m, p, s)
    return output
