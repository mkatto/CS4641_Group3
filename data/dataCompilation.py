import pandas as pd
import numpy as np

xyear = 2007
yyear = 2010
#countries = ['AND','AUT','BEL','CHE','CHI','CYP','CZE','DEU','DNK','ESP','EST','FIN','FRA','FRO','GBR','GIB','GRC','GRL','HRV','HUN','IMN','IRL','ISL','ITA','LIE','LTU','LUX','LVA','MCO','NLD','NOR','POL','PRT','SMR','SVK','SVN','SWE']


interest = pd.read_csv("realinterestrates.csv").set_index('Country Name')
# https://data.worldbank.org/indicator/SL.UEM.TOTL.NE.ZS

interest = pd.read_csv("lendinginterestrates.csv").set_index('Country Name')
# https://data.worldbank.org/indicator/FR.INR.LEND

minwage = pd.read_csv("ilostatminwage.csv")#
minwage = minwage.loc[minwage['time'] == xyear].loc[minwage['classif1'] == 'CUR_TYPE_USD'].set_index('ref_area.label')
print(minwage['obs_value'])
# https://www.ilo.org/shinyapps/bulkexplorer40/?lang=en&segment=indicator&id=EAR_4MMN_CUR_NB_A

capitalism = pd.read_csv("capitalism.csv").set_index('country')
# https://worldpopulationreview.com/country-rankings/capitalist-countries

education = pd.read_csv("education.csv").set_index('Country Name')
# https://data.worldbank.org/indicator/SE.XPD.TOTL.GB.ZS

tax = pd.read_csv("taxes.csv").set_index('Country Name')
# https://data.worldbank.org/indicator/GC.TAX.YPKG.ZS

unemployment = pd.read_csv("unemployment.csv").set_index('Country Name')
# https://data.worldbank.org/indicator/FR.INR.RINR
df1 = pd.DataFrame({'country': minwage.index
                    }).set_index('country')
for con in minwage.index:
    print(con)
    #print(df1.loc[con, 'interest'])
    try:
        df1.loc[con, 'interest'] = interest.loc[con, str(xyear)]
        df1.loc[con, 'tax'] = tax.loc[con, str(xyear)]
        df1.loc[con, 'unemployment'] = unemployment.loc[con, str(yyear)]
        df1.loc[con, 'education'] = education.loc[con, str(xyear)]
    except:
        print("F")
    try:
        df1.loc[con, 'minwage'] = minwage.loc[con, 'obs_value']
        df1.loc[con, 'minwage2'] = minwage.loc[con, 'obs_value'] * minwage.loc[con, 'obs_value']
    except:
        print("F")
    try:
        df1.loc[con, 'capitalism'] = capitalism.loc[con, 'economicFreedomScore']
    except:
        print("F")
print(df1)
print("education", df1["education"].isna().sum())
print("Interest", df1["interest"].isna().sum())
print("Unempl", df1["unemployment"].isna().sum())
print("tax", df1["tax"].isna().sum())
print("minwage", df1["minwage"].isna().sum())
df1 = df1.dropna()
print(df1)

import statsmodels
from statsmodels.formula.api import ols
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt
mod = sm.formula.ols(formula='unemployment ~ interest + tax + minwage + education + capitalism', data=df1)
mod = sm.formula.ols(formula='unemployment ~ minwage + capitalism', data=df1)
res = mod.fit()
print("Res Summary \n", res.summary(), "\n")
print("ANOVA: \n", sm.stats.anova_lm(res, typ=2))
print("equation: \n", res.params[1], "x +", res.params[0])
print(df1.corr())