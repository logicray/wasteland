import random

wfile = open('test.txt', 'w')
for i in range(1000):
    x = random.uniform(0,1000)
    y = random.uniform(0,1000)
    wfile.write(str(x))
    wfile.write('\t\t')
    wfile.write(str(y))
    wfile.write('\n')

