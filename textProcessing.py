import numpy as np

text_file = open("adr2b-Ys-v3.txt", "r")
line=text_file.read().split('\n')

#print(line)
#print(line[0])  #Percentage%          0.5         1.0         5.0         10.0    Y
#print(line[1])  #Upper value ITQ:    7       11       35       63       Y
a = line[2]
print(a)
b = a.split('[')
print(b[0])	#has the conformation timestamp
d = b[0].split('at-frame')
print(d)
e = d[1].split('\t') #we need e[0]. This contains just the conformation timestamp with a right space.
# print(e)
# f = e[0].split(' ')
# print(f)
c = b[1].split(']')
print(b[1])
#print(c[0]) #not required
print(c[1])	#has the Yes label for the conformations
print(type(e[0]))

#create a dataframe or dictionary with conformation timestamp and Yes label

#compare the timestamps from the .csv file with the dictionary entries and add the label accordingly