import pandas as pd

print 0
a = pd.read_csv("AddWSandBoro/rentList_All.csv")
print 1
b = pd.read_csv("ward-profiles-excel-version.csv")
print 2
c = pd.read_csv("london-borough-profiles.csv")
print 3

d = a.merge(b, left_on='ward_code', right_on='New code')
merged = d.merge(c, left_on='borough_code', right_on='New code')
merged.to_csv("rentList_All_merged.csv", index=False)