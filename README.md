# Georgia Tech CS 4641 - Machine Learning Group 3 Project
## Analyzing Unemployment
Saikanam Siam, Matthew Oswald, Sanjit Kumar, Matthew K Attokaren

## Motivation
We are currently experiencing the second major economic recession of our lifetimes due to the COVID-19 pandemic. In the age of information, the advancement of machine learning algorithms has allowed us to gain more knowledge from global economic data than ever before [1]. Integrating data analysis with financial data has made tremendous strides in trading, fraud detection, and market forecasting [2]. These same techniques can be applied on large scale macroeconomic data for countries to find patterns between economic policy and statistics, and unemployment [3].  

The unemployment rate is a common metric used to describe a recession; however, a recession affects many more aspects of a country's economy. Our proposal is to use publicly available datasets to collect yearly attributes such as deficit, tax rates, interest rates, minimum wage, etc. and see if there is any clear correlation with the unemployment rate. Our main goal is to identify which factors or policies contribute to an increase in unemployment for countries.

The problem we are trying to solve is how can unemployment rates be predicted using publicly available data of countries’ yearly economic attributes (such as property rights, government integrity, tax burden, etc.) from the past decade? 

### Why is it important and why should we care? 
Economic recessions can have a major negative impact on countries and their citizens.  If we can learn which combination of economic indicators can predict a recession and increased unemployment, then governments and people can use this information to better prepare for a downturn and inform economic policy makers. 

## The Dataset
We got our data set from the [Heritage Foundation](https://www.heritage.org/index/explore?view=by-region-country-year&u=637302137906965144).  The dataset contains thirteen indexes of economic freedom for each year.  These indices include Property Rights, Judicial Effectiveness, Government Integrity, Tax Burden, Government Spending, Fiscal Health, Business Freedom, Labor Freedom, Monetary Freedom, Trade Freedom, Investment Freedom and Financial Freedom. The Heritage Foundation records these indexes from over 150 countries, with some indexes dating back to 1995 for some of the countries.

![Heritage Foundation Dataset](HeritageFoundation2006_2020overallscore.jpg)

We gathered data on multiple facets of economics that covers economics at a macro level and micro level instead of looking at them individually. Having so many different features reduces the risk of confounding variable skewing our dataset and us not being able to identify them, although this causes one of the primary problems with high-dimensionality also known as model overfitting. This is where PCA comes in. 


### Method
First we look at a specific economic policy that was historically suggested and used during the first great depression, known as Keynesian policies. According to Keynesian Economics, there are three main ways a government can stabilize an economy during a recession. One way is to decrease interest rates. This allows easier borrowing for businesses and therefore creates more jobs. The second way a government can increase jobs is by decreasing taxes. This is based on the same idea that additional money in the hands of citizens will allow for more businesses to stay open. The third main way a government can stabilize a recession is by increasing spending to again put more money in the hands of citizens.

![heatmap](MacroPolicy/heatmap.png)

We can notice a couple of basic correlations on the preliminary heatmap. For example, we can see that obviously the more farming an economy has, the fewer service employees the country has. Then we will start to notice other things like the higher the minimum wage, the more energy costs. We can see that the interest, taxes, and unemployment before the recession are correlated to those variables during the recession. The correlations to unemployment are:


![heatmap2](MacroPolicy/heated.PNG)

Creating a Model to decide Changes
One major issue that our team had was that there is a very limited amount of consistent data for the countries. Of the features that we had, the maximum number of data points was 162 and the minimum was 80. This created a necessity to make our model as simple as possible as including all potential features would limit our data to only 23 data points. To ensure we only kept only the most important variables, we created a function much like regsubsets where it chooses the highest R^2 score from possible subsets and we are able to force certain features to be included in R except it accounted for potential blank data points for each combination as opposed to the whole dataset (Our function only seemed efficient enough to work on up to around 200 combinations).

Variables that don't affect future unemployment: Initial Government Spending (prespending), Change in Real Interest Rates (realinterest), Real Interest Rates (prerealinterest), and Change in Taxation Rates (tax).
All of these had P>|t| of .4 or more.
Our final model for identifying changes in policy looked like this:

![prediction](MacroPolicy/predictols.PNG)

Based on this model, we can choose two policies between monetary and fiscal policy.
Decrease Spending During a Recession: We see that there is no impact on unemployment based on initial spending, but we see that there is a positive correlation between the act of increasing spending and unemployment. This goes against Keynesian economics, but a possible explanation would be that increased spending makes people wary of the possibility of a country going bankrupt and therefore less advantageous for business.
Make Taxes on income, profits and capital gains 42% Before the Next Recession: We can see that the 2007 tax variable is -.0757X+.0009X^2 If we solve for the minimum, we can see that we can decrease unemployment by up to 1.6% by putting the taxes to 42%.

### Results

To evaluate our approach, grouped our data into two training sets.  We wanted to see if we could use a countries previous years' data to predict a future year's unemployment rate.  We also wanted to see if we could predict one countries unemployment rate by using other countries' data.

### Conclusion

### References
- [1] Athey, S. (2018). The impact of machine learning on economics. In The economics of artificial intelligence: An agenda (pp. 507-547). University of Chicago Press. 

- [2] Puglia, M., & Tucker, A. (2020). Machine Learning, the Treasury Yield Curve and Recession Forecasting. 

- [3] Katris, C. (2020). Prediction of unemployment rates with time series and machine learning techniques. Computational Economics, 55(2), 673-706. 

- [4] Jahan, S., Mahmud, A. S., & Papageorgiou, C. (2014). What is Keynesian economics?. International Monetary Fund, 51(3). 
