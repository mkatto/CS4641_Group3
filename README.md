## Georgia Tech CS 4641 - Machine Learning Group 3 Project
### Analyzing Unemployment
Saikanam Siam, Matthew Oswald, Sanjit Kumar, Matthew K Attokaren

### Motivation
We are currently experiencing the second major economic recession of our lifetimes due to the COVID-19 pandemic. In the age of information, the advancement of machine learning algorithms has allowed us to gain more knowledge from global economic data than ever before [1]. Integrating data analysis with financial data has made tremendous strides in trading, fraud detection, and market forecasting [2]. These same techniques can be applied on large scale macroeconomic data for countries to find patterns between economic policy and statistics, and unemployment [3].  

The unemployment rate is a common metric used to describe a recession; however, a recession affects many more aspects of a country's economy. Our proposal is to use publicly available datasets to collect yearly attributes such as deficit, tax rates, interest rates, minimum wage, etc. and see if there is any clear correlation with the unemployment rate. Our main goal is to identify which factors or policies contribute to an increase in unemployment for countries. 

### Dataset
We got our data set from the [Heritage Foundation](https://www.heritage.org/index/explore?view=by-region-country-year&u=637302137906965144).  The dataset contains thirteen indices of economic freedom for each year.  These indices include Property Rights, Judicial Effectiveness, Government Integrity, Tax Burden, Government Spending, Fiscal Health, Business Freedom, Labor Freedom, Monetary Freedom, Trade Freedom, Investment Freedom and Financial Freedom. 
![Heritage Foundation Dataset](HeritageFoundation2006_2020overallscore.jpg)
We gathered data on multiple facets of economics that covers economics at a macro level and micro level instead of looking at them individually. Having so many different features reduces the risk of confounding variable skewing our dataset and us not being able to identify them, although this causes one of the primary problems with high-dimensionality also known as model overfitting. This is where PCA comes in. 

### Results

### Conclusion

### References
[1] Athey, S. (2018). The impact of machine learning on economics. In The economics of artificial intelligence: An agenda (pp. 507-547). University of Chicago Press. 
[2] Puglia, M., & Tucker, A. (2020). Machine Learning, the Treasury Yield Curve and Recession Forecasting. 
[3] Katris, C. (2020). Prediction of unemployment rates with time series and machine learning techniques. Computational Economics, 55(2), 673-706. 
[4] Jahan, S., Mahmud, A. S., & Papageorgiou, C. (2014). What is Keynesian economics?. International Monetary Fund, 51(3). 
