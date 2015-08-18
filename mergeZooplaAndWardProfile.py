import pandas as pd

a = pd.read_csv("AddWSandBoro/HousePriceList_All_Boro_WS.csv")
b = pd.read_csv("ward-profiles-trimmed.csv")
c = pd.read_csv("london-borough-profiles-trimmed.csv")

d = a.merge(b, left_on='ward_code', right_on='New code')
merged = d.merge(c, left_on='borough_code', right_on='New code')
merged.to_csv("HousePriceList_All_Boro_WS_BoroWard.csv", index=False)