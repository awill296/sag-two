"""
#. Introductory Video to v1: < https://youtu.be/2sYbTEF___c >

#. How to Run this App:
00. Open Terminal
01. Cast "python capcartsag.py"
02. Open browser to http://localhost:8889

Note: The file backdraft.ipynb is for logic function development purposes and is not integrated with the app itself.

#. Intended Application Flow:
00. System loads data frames from csvs in datasources
01. System displays dropdown of Response variables
02. User selects desired Response variable
03. User clicks first Submit button
04. System displays checklist of Predictor variables
05. User selects desired Predictor variables
06. User clicks second Submit button
07. System displays checklist of Measures
08. User selects desired Measures
09. User clicks third Submit button
10. System displays dropdown of Propositions
11. User selects desired Proposition
12. User clicks fourth Submit button
13. System displays set of Model Configuration Settings
14. User sets desired Model Configuration values
15. User clicks final Submit button
16. System performs calculations
17. System displays histogram of chosen Response variable
18. System displays histogram of chosen Predictor variable(s)
19. System displays chosen Measures for all variables
20. System displays details of chosen Proposition

Note: Selection of Predictor variables and Measures is optional. 
	If user has not selected any Measures, none will be displayed.
	If user has not selected any Predictor variables, 
		the System will display the requested Measures 
		for the Response variable alone and not calculate a Proposition.

#. Available Items as of v1.5:
00. Response Variables: NDS and Subscores 
			(Behavior, Brainstem_Function, General_Behavioral
            , Motor_Assessment, Motor_Behavior, Seizures, Sensory_Assessment
			, High/Low binary forms of each where applicable)
01. Predictor Variables: HR @20,40,60 min; MAP @20,40,60 min; 
		Burst Period Start, End, and Duration; Burst Count, Duration Max, Duration Percent, Duration Max Percent, High/Low binary forms for each
        Electrode {1,2}>{Power,Shannon Entropy,Tsallis Entropy,Power Fraction,Shannon Entropy Fraction, Tsallis Entropy Fraction}
            >{Delta,Theta,Alpha,Beta,Gamma,SuperGamma}>{AUC,IQR,MAD,Max,Mean,Median,Min,Range,Spot,STD}>@20,40,60Mins
02. Measures: 
			Bivariate: Chi-Squared Test of Independence of Categorical Variables
						, Spearman's Rank Correlation Coefficient,  
			Univariate: IQR, Kurtosis, Max, Mean, Min, Mode, Range, Skew
						Median Absolute Deviation (median of abs dev from median)
						Relative Standard Deviation ( std/mean )
03. Propositions: Lasso Regression, Linear Discriminant Analysis, Linear Regression, Logistic Regression
					Naive Bayes (Gaussian), Neural Network, Support Vector Machine
					Note: LDA, LinReg, LogReg, and SVM have automated iterative feature selection
05. Model Configuration Settings: 
	All: Random Seed, Percent of Data to use in Training vs Testing 
	Lasso Regression: Inclusion of Intercept, Magnitude of Iteration Limit, Tolerance of Learning Rate
	Linear Discriminant Analysis: Binarization Threshold Percent, Tolerance of Learning Rate
	Linear Regression: Inclusion of Intercept
	Logistic Regression: Inclusion of Intercept, Magnitude of Iteration Limit, Binarization Threshold Percent 
	Naive Bayes (Gaussian): Binarization Threshold Percent
	Neural Network: Iteration Limit, Number of Layers, Nodes per Layer
	Support Vector Machine: Inclusion of Intercept, Magnitude of Iteration Limit, Binarization Threshold Percent, Tolerance of Learning Rate
06. Proposition Details:
	All: Error Rate on Training and Test data, ROC Curve on Training and Test data  
	Neural Network: Weights
	Naive Bayes: Theta, Variance, Epsilon
	All others: Coefficient and Intercept values
	
Warning: Due to the low sample size of data available in v1.5, changes to the Random Seed, 
	Percent Training Data, and Binarization Threshold Percent may cause errors 
	due to insufficient data variation. 

#. Items in-progress for v2:
00. Display: Scatter plot(s) for Proposition:Linear Regression, Proposition:LDA, Proposition:SVM
            , Node-Graph for Proposition:Decision Tree and Proposition:Neural Network
            , Univariate: Box-and-whisker plot
            , Bivariate: Scatter Plots
00. Add setting to choose graph display: Histograms, Box-whisker, Scatter Plots, or None
00. Add model configuration setting to change limit on iterative feature selection
00. Add setting to make iterative feature selection optional
00. Applying Measures and Propositions only with appropriate variable Level of Measurement 
00. Predictor variables listed only when causally or logically valid for chosen Response variable
00. Expand list of available Response variables
00. Measures: Covariance, Normality (display Jarque-Bera, Lilliefors, and Shapiro-Wilk results on all variables) 
00. Propositions: Decision Tree, Naive-Bayes (Categorical)

#. Items Planned for future versions:
v3. create executable, time series analysis, functions of data as predictor and response variables (e.g. entropy), 
	, more model configuration settings (feature select)
	, proposition tutorials (feature request), model R^2 (feature request), experimental cohort selection (feature request)
v4 (draft). replace csvs with database of experimental data, ability to download results, add login security (feature request)
v5 (draft). additional validation methods, additional propositions, model stacking

#. Items specifically not Planned for future versions:
	Cramer von Mises, Cohen's Kappa, Poisson Regression
"""