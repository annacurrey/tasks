

import numpy as np

#thisisanexample
#100010101000000
# this is an example

def convertToLabels(string):
    res=[[1,-1]]
    i=1
    while i < len(string):
        if string[i] == " ":
            res.append([1,-1])
            i+=2
        else:
            res.append([-1,1])
            i+=1
    result=np.array(res,dtype="float")
    return result


def create_char_dict(f):
    char_dict=dict()
    with open(f) as file:
        chars=set()
        text=file.read()
        for char in text:
            chars.add(char)


    base=[-1 for i in range(len(chars))]
    i=0
    for char in chars:
        vec=np.array(base, dtype="float")
        vec[i]=1.
        char_dict[char]=vec
        i+=1

    return char_dict


def decode_line(unsplitted,labels):
    res = ""
    for i in range(0, len(labels)):
        if labels[i][0]>labels[i][1]:
            res += ' '
        res += unsplitted[i]
    return res    

def translate_line(string, char_dict):
    result=[]
    for c in string:
        if c != " ":
            result.append(char_dict[c])

    return result

if __name__=="__main__":
    print(convertToLabels("this is an example"))
