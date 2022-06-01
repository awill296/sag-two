"""
#. Introductory Video: < https://youtu.be/2sYbTEF___c >

#. How to Run this App:
00. Open Terminal
01. Cast "python capcartsag.py"
02. Open browser to http://jupyter.biostat.jhsph.edu:[X]/, [X] is your os.getuid()+30

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
		for the Predictor variable alone and not calculate a Proposition.

#. Available Items in v1:
00. Response Variables: NDS, HR @60 min, MAP @60min
01. Predictor Variables: HR @20 min, HR @40min, MAP @20min, MAP @40min
02. Measures: Mean, Standard Deviation, Spearman's Rank Correlation Coefficient
03. Propositions: Linear Regression, Logistic Regression, Neural Network
05. Model Configuration Settings: 
	All: Random Seed, Percent of Data to use in Training vs Testing 
	Linear Regression: Inclusion of Intercept
	Logistic Regression: Inclusion of Intercept, Magnitude of Iteration Limit, Binarization Threshold Percent 
	Neural Network: Iteration Limit, Number of Layers, Nodes per Layer
06. Proposition Details:
	All: Error Rate on Training and Test data, ROC Curve on Training and Test data  
	Linear Regression: Coefficient and Intercept values
	Logistic Regression: Coefficient and Intercept values 
	Neural Network: Weights

Warning: Due to the low sample size of data available in v1, changes to the Random Seed, 
	Percent Training Data, and Binarization Threshold Percent may cause errors 
	due to insufficient data variation. 
	
#. Items drafted but not yet implemented in v1:
00. Applying Measures and Propositions only with appropriate variable Level of Measurement 
01. Measure: Pearson Correlation Coefficient (need a higher sample size before this would be valid)
02. Proposition: Decision Tree (allows for inferences on Nominal data and has high interpretability)
03. Display: Scatter plot for Proposition:Linear Regression, Node-Graph for Proposition:Neural Network 

#. Items Planned for future versions:
v2. higher sample size, additional predictor and response variables, v1 backlog (see above)
v3. time series analysis, functions of data as predictor and response variables (e.g. entropy), more model configuration settings
	, proposition tutorials (feature request), experimental cohort selection (feature request), model R^2 (feature request)
v4 (draft). replace csvs with database of experimental data, ability to download results, add login security (feature request)
v5 (draft). additional validation methods, additional propositions, model stacking
"""