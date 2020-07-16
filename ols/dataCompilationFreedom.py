import pandas as pd
import numpy as np
#################
yyear = 2008
#################

df1 = pd.read_csv("freedom.csv")
# https://www.heritage.org/index/explore?view=by-region-country-year&u=637302137906965144

unemployment = pd.read_csv("unemployment.csv").set_index('Country Name')[str(yyear)]
# https://data.worldbank.org/indicator/FR.INR.RINR

df1 = pd.merge(df1, unemployment, right_on = 'Country Name', left_on= 'Name')
for name in df1.keys():
    df1 = df1.rename(columns= {name: name.replace(" ", "")})
df1 = df1.rename(columns= {str(yyear): "unemployment"})
df1 = df1.dropna()
print(df1)

import statsmodels
from statsmodels.formula.api import ols
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt

print(df1.corr().loc["unemployment"])
mod = sm.formula.ols(formula='unemployment ~ PropertyRights + GovernmentIntegrity + TaxBurden + GovernmentSpending + BusinessFreedom + LaborFreedom + MonetaryFreedom + TradeFreedom + InvestmentFreedom + FinancialFreedom', data=df1)
#InvestmentFreedom
mod = sm.formula.ols(formula='unemployment ~ PropertyRights + GovernmentIntegrity + TaxBurden + GovernmentSpending + BusinessFreedom + LaborFreedom + MonetaryFreedom + TradeFreedom + FinancialFreedom', data=df1)
#TradeFreedom
mod = sm.formula.ols(formula='unemployment ~ PropertyRights + GovernmentIntegrity + TaxBurden + GovernmentSpending + BusinessFreedom + LaborFreedom + MonetaryFreedom + FinancialFreedom', data=df1)
#GovernmentSpending
mod = sm.formula.ols(formula='unemployment ~ PropertyRights + GovernmentIntegrity + TaxBurden + BusinessFreedom + LaborFreedom + MonetaryFreedom + FinancialFreedom', data=df1)
#LaborFreedom
mod = sm.formula.ols(formula='unemployment ~ PropertyRights + GovernmentIntegrity + TaxBurden + BusinessFreedom + MonetaryFreedom + FinancialFreedom', data=df1)
#FinancialFreedom
mod = sm.formula.ols(formula='unemployment ~ PropertyRights + GovernmentIntegrity + TaxBurden + BusinessFreedom + MonetaryFreedom', data=df1)
#PropertyRights
#Using this model, there is an 80% chance that more property rights is helpful during a recession.
mod = sm.formula.ols(formula='unemployment ~ GovernmentIntegrity + TaxBurden + BusinessFreedom + MonetaryFreedom', data=df1)
#TaxBurden
#Using this model, there is an 85% chance that higher marginal tax rates are helpful during recessions
mod = sm.formula.ols(formula='unemployment ~ GovernmentIntegrity + BusinessFreedom + MonetaryFreedom', data=df1)
#BusinessFreedom
mod = sm.formula.ols(formula='unemployment ~ GovernmentIntegrity + MonetaryFreedom', data=df1)
#Monetary Freedom is the measure of inflation in the three years prior to 2007.
#   We can see that countries who had inflation under control had higher unemployment in a weird twist!!
#We can see that countries who have high curruption do worse in recessions.
res = mod.fit()
print("Res Summary \n", res.summary(), "\n")
print("ANOVA: \n", sm.stats.anova_lm(res, typ=2))
print("equation: \n", res.params[1], "x +", res.params[0])
