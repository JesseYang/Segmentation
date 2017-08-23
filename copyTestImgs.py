import shutil

f = open('goose_test.txt', 'r')

for i in f.readlines():
    shutil.copy(i[:-1], 'testImgs/%s'%(i[:-1].split('/')[-1]))