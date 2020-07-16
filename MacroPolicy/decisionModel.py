import pandas as pd
import numpy as np
import statsmodels
from statsmodels.formula.api import ols
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt
from itertools import permutations
from itertools import combinations
import operator

df1 = pd.read_csv("compilationI.csv")

print()
corr = df1.corr()

myBasicCorr = df1.corr()
plts = sns.heatmap(df1.corr().loc["postunemployment"].to_frame(), annot = True, cmap='coolwarm', xticklabels=True, yticklabels=True)
plt.show(plts)

df1['agriculture2']=df1['agriculture']*df1['agriculture']
df1['services2']=df1['services']*df1['services']
df1['industry2']=df1['industry']*df1['industry']
df1['minwage2']=df1['minwage']*df1['minwage']
df1['tax2']=df1['tax']*df1['tax']
df1['pretax2']=df1['pretax']*df1['pretax']
df1['preunemployment2']=df1['preunemployment']*df1['preunemployment']
df1['realinterest2']=df1['realinterest']*df1['realinterest']
df1['kWhcapita2']=df1['kWhcapita']*df1['kWhcapita']
df1['incomeshare2']=df1['incomeshare']*df1['incomeshare']
df1['spending2']=df1['spending']*df1['spending']
df1['prespending2']=df1['prespending']*df1['prespending']
df1['prerealinterest2']=df1['prerealinterest']*df1['prerealinterest']
df1['realinterest2']=df1['realinterest']*df1['realinterest']
df1['taxag']=df1['agriculture']*df1['tax']


var=['agriculture','services','preunemployment','kWhcapita','incomeshare','agriculture2','services2','preunemployment2','kWhcapita2','incomeshare2']
#These are variables chosen from for vars additional variables
var=['agriculture','services','preunemployment','kWhcapita','agriculture2','services2','preunemployment2','kWhcapita2']

def fitwithvar(df, vars = 7, var = var):
    perm = combinations(var, vars)
    dic = {}
    #These are the variables that are forced into the formula
    addvar = ['spending', 'spending2','pretax', 'pretax2']
    for currperm in list(perm):
        modfor = str("postunemployment") + "~"
        for v in currperm:
            df2 = df.dropna(subset=[v])
            modfor += v + "+"
        for v in addvar:
            df2 = df.dropna(subset=[v])
            modfor += v + "+"
        modfor = modfor[:-1]
        mod = sm.formula.ols(formula=modfor, data=df2)
        res = mod.fit()
        dic[res.rsquared] = modfor
    mxrsqrd = max(dic.keys())
    mod = sm.formula.ols(formula=dic[mxrsqrd], data=df2)
    res = mod.fit()
    print("Res Summary \n", res.summary(), "\n")
#fitwithvar(df = df1, vars = 6)
#fitwithvar(df = df1, vars = 5)
fitwithvar(df = df1, vars = 4)
fitwithvar(df = df1, vars = 3)
fitwithvar(df = df1, vars = 2)
fitwithvar(df = df1, vars = 1)


