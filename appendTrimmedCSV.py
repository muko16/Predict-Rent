fout = open("rentList_All_final.csv","a")

num_line = 0

# first file:
#for line in open("rentList_NW_WS_BoroWard-trimmed.csv"):
#    fout.write(line)
#    num_line += 1

#print num_line

# rest:   
# e = open("rentList_NW_WS_BoroWard-trimmed.csv")
# e.next() # skip the header
# num_line = 0
# for line in e:
#     fout.write(line)
#     num_line += 1
# e.close() # not really needed
# print num_line

# f = open("rentList_SE_WS_BoroWard-trimmed.csv")
# f.next() # skip the header
# num_line = 0
# for line in f:
#     fout.write(line)
#     num_line += 1
# f.close() # not really needed
# print num_line

# g = open("rentList_WC_WS_BoroWard-trimmed.csv")
# g.next() # skip the header
# num_line = 0
# for line in g:
#     fout.write(line)
#     num_line += 1
# g.close() # not really needed
# print num_line

# rest:   
f = open("rentList_SW_WS_BoroWard-trimmed.csv")
f.next() # skip the header
num_line = 0
for line in f:
    fout.write(line)
    num_line += 1
f.close() # not really needed
print num_line

g = open("rentList_W_WS_BoroWard-trimmed.csv")
g.next() # skip the header
num_line = 0
for line in g:
    fout.write(line)
    num_line += 1
g.close() # not really needed
print num_line

fout.close()
