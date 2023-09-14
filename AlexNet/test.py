import os

a = os.getcwd()
b = os.path.join(os.getcwd(), "../..")
c = os.path.abspath(os.path.join(os.getcwd()))

l = [a, b, c]

for i in l:
    print(i)
