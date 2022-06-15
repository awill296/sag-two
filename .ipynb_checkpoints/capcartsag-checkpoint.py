from collections import OrderedDict; from dash import ALL, dcc, html, Input, MATCH, Output, State;
from flask import Markup; from IPython.display import display, Markdown; from itertools import compress;
from matplotlib.ticker import MaxNLocator; from plotly.subplots import make_subplots;
from plotly.tools import mpl_to_plotly; from sklearn import svm,tree; from sklearn.feature_selection import RFECV;
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc; 
from scipy.stats import chi2_contingency,iqr,kurtosis,median_abs_deviation,mode,pearsonr,spearmanr,skew; 
from statistics import mode; from tensorflow.keras import layers, models, losses; 
from torch.utils.data import TensorDataset, DataLoader;
import dash; import dash_useful_components as duc; import graphviz; import math; import matplotlib.pyplot as plt; 
import networkx as nx; import numpy as np; import numpy.random as rnd; import pandas as pd;
import os; import plotly.express as px; import plotly.graph_objects as go; import requests as rq;
import seaborn as sns; import sklearn as skl; import sklearn.linear_model as lm; 
import sklearn.naive_bayes as nb; import sklearn.discriminant_analysis as lda; 
import statistics as stttx; import statsmodels as sm; import statsmodels.api as sma; 
import statsmodels.formula.api as smf; import requests as rq; import tensorflow as tf; 
import torch; import torch.nn as nn; import torch.nn.functional as F; import torch.optim as optim; 
import torchvision; import torchvision.transforms as transforms; import urllib.request; import warnings;

# file settings
warnings.simplefilter(action='ignore', category=Warning); sns.set(); np.set_printoptions(threshold=np.inf); 
pd.set_option('display.max_rows',999); pd.set_option('display.max_columns', 12);  
pd.set_option('display.max_rows',999); pd.set_option('display.width', 200);
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3';

# global settings
MRL = 3; #Measure Rounding Level
folder = "datasources"; files = ["cohorts.csv","datafile.csv","measures.csv","propositions.csv","schema.csv"]
subtitles = {'Title1':"Thakor Lab",
			'Title2':"Cerebrovascular Autoregulation and Post-Cardiac Arrest Resuscitation Therapies Team",
			'Title3':"Statistical Analysis GUI, v1.0",
			'Step00':"Select Response Variable: ", 'Step04':"Check Predictor Variable(s) to test: ",
			'Step06':"Select Measures to Display: ", 'Step07':"Select Model Proposition to Calculate: ",
			'Step09':"Enter the Configuration Settings for the selected model (default values pre-entered): ",
			'Step09.01':"Random Seed: ", 'Step09.02':"Percent of Data to use in Training vs Testing: ",
			'Step09.03':"Include Intercept value in Model: ", 'Step09.04':"Magnitude of Iteration Limit: ",
			'Step09.05':"Percentile to use as Threshold for Binarization of Response Variable: ", 
			'Step09.06':"Number of Layers in Model: ", 'Step09.07':"Branches or Nodes per Layers of Model: ",
			'Step09.08':"Learning Rate of Model: ",
			'Step12.01':"Histogram of Response Variable Values", 'Step12.02':"Histogram(s) of Predictor Variable Values",
			'Step13':"Table of Requested Measure(s)",'Step14':"Details of Requested Model"};

# globalized variables:
# df_DataProc; df_Schema; df_Meas; df_Prop;
# namesPred; namesResp; namesMeas; namesProp;
# uniList

app = dash.Dash(__name__, suppress_callback_exceptions=True);

def serve_layout():
	loadData();
	return html.Div([
		#html.H2(subtitles['Title1']),
		#html.H2(subtitles['Title2']),
		html.H2(subtitles['Title3']),
		html.Div(style={'visibility': 'visible'},children=[
			html.H3(subtitles['Step00']),
			dcc.Dropdown( options= [{'label': v, 'value': k} for k, v in namesResp.items()]
						 ,id={'type':'selector-uni', 'role':'choice', 'index': 'resp'},value = 1, disabled = False),
			html.Button('Submit', id={'type':'button', 'role': 'submit', 'index': 'resp'}, n_clicks=0)
		], id={'type':'div', 'role': 'prompt', 'index': 'init'}),
		html.Div(style={'visibility':'hidden'}, children=[
				html.H3(subtitles['Step04']),
				html.Div(duc.CheckBoxTree(id={'type':'selector-tree', 'role':'choice', 'index': 'pred'}
											,nodes=genCBTPred(), showNodeIcon=False)),
				#dcc.Checklist(options= [{'label': v, 'value': k} for k, v in namesPred.items()]
							  #,id={'type':'selector-multi', 'role':'choice', 'index': 'pred'}),
				html.Button('Submit', id={'type':'button', 'role': 'submit', 'index': 'pred'}, n_clicks=0)
		], id={'type':'div', 'role': 'prompt', 'index': 'resp'}),
		html.Div(style={'visibility':'hidden'}, children=[
			html.H3(subtitles['Step06']),
			html.Div(duc.CheckBoxTree(id={'type':'selector-tree', 'role':'choice', 'index': 'meas'}
											,nodes=genCBTMeas(), showNodeIcon=False)),
			#dcc.Checklist(options= [{'label': v, 'value': k} for k, v in namesMeas.items()]
						  #,id={'type':'selector-multi', 'role':'choice', 'index': 'meas'},inline=False),
			html.Button('Submit', id={'type':'button', 'role': 'submit', 'index': 'meas'}, n_clicks=0)
		], id={'type':'div', 'role': 'prompt', 'index': 'pred'}),
		html.Div(style={'visibility':'hidden'}, children=[
			html.H3(subtitles['Step07']),
			dcc.Dropdown( options= [{'label': v, 'value': k} for k, v in namesProp.items()]
						 ,id={'type':'selector-uni', 'role':'choice', 'index': 'prop'},value = 2),
			html.Button('Submit', id={'type':'button', 'role': 'submit', 'index': 'prop'}, n_clicks=0)
		], id={'type':'div', 'role': 'prompt', 'index': 'meas'}),
		html.Div([
				html.Table([
					html.Tbody([
						html.Tr([
							html.Td([
								html.Div(style={'visibility':'inherited'}, children=[
									html.Span(subtitles['Step09.01']),
									dcc.Input(type='number',min=1,step=1,value=123
												   , id = {'type':'selector-uni', 'role':'conf', 'index': 'seed'})
									], title='Seed', id={'type':'div', 'role': 'conf-any', 'index': 'seed'}),
							]),
							html.Td([
								html.Div(style={'visibility':'hidden'}, children=[
									html.Span(subtitles['Step09.05']),
									dcc.Slider(10,90,10,value=50
											   ,id={'type':'selector-uni', 'role':'conf', 'index': 'binthresh'}),
									], title='BinThresh', id={'type':'div', 'role': 'conf', 'index': 'binthresh'}),
							])
						]),
						html.Tr([
							html.Td([
								html.Div(style={'visibility':'inherited'}, children=[
									html.Span(subtitles['Step09.02']),
									dcc.RadioItems(options = [{'label': '90', 'value': 90},{'label': '75', 'value': 75}
															  ,{'label': '50', 'value': 50},{'label': '25', 'value': 25}
															  ,{'label': '10', 'value': 10}],value = 50
												   , id = {'type':'selector-uni', 'role':'conf', 'index': 'trainPct'})
									], title='TrainPercent', id={'type':'div', 'role': 'conf-any', 'index': 'trainPct'}),
							]),
							html.Td([
								html.Div(style={'visibility':'hidden'}, children=[
									html.Span(subtitles['Step09.06']),
									dcc.Slider(1,10,1,value=1
											   ,id={'type':'selector-uni', 'role':'conf', 'index': 'layers'}),
									], title='Layers', id={'type':'div', 'role': 'conf', 'index': 'layers'}),
							])
						]),
						html.Tr([
							html.Td([
								html.Div(style={'visibility':'hidden'}, children=[
									html.Span(subtitles['Step09.03']),
									dcc.RadioItems(options = [{'label': 'Yes', 'value': 1},{'label': 'No', 'value': 0}],value = 1
												   , id = {'type':'selector-uni', 'role':'conf', 'index': 'intercept'})
									], title='Intercept', id={'type':'div', 'role': 'conf', 'index': 'intercept'}),
							]),
							html.Td([
								html.Div(style={'visibility':'hidden'}, children=[
									html.Span(subtitles['Step09.07']),
									dcc.Slider(2,16,1,value=2
											   ,id={'type':'selector-uni', 'role':'conf', 'index': 'nodes'}),
									], title='Nodes', id={'type':'div', 'role': 'conf', 'index': 'nodes'}),
							])
						]),
						html.Tr([
							html.Td([
								html.Div(style={'visibility':'hidden'}, children=[
									html.Span(subtitles['Step09.04']),
									dcc.Slider(0,9,1,value=2
											   ,id={'type':'selector-uni', 'role':'conf', 'index': 'magnitude'}),
									], title='Magnitude', id={'type':'div', 'role': 'conf', 'index': 'magnitude'}),
							]),
							html.Td([
								html.Div(style={'visibility':'hidden'}, children=[
									html.Span(subtitles['Step09.08']),
									dcc.Slider(0,9,1,value=2
											   ,id={'type':'selector-uni', 'role':'conf', 'index': 'learnrate'}),
									], title='LearnRate', id={'type':'div', 'role': 'conf', 'index': 'learnrate'}),
							])
						])
					])
				]),
				html.Button('Submit', id={'type':'button', 'role': 'submit', 'index': 'conf'}, n_clicks=0)
		], id={'type':'div', 'role': 'prompt', 'index': 'prop'}),
		html.Br(),
		html.Br(),
		html.Br(),
		html.Div(style={'visibility':'hidden'}, children=[
			html.Div([
					html.H2(subtitles['Step12.01']),
					dcc.Graph(id={'type':'graph', 'role': 'result', 'index': 'resp'}),
					], id={'type':'div', 'role': 'result', 'index': 'resp'}),
			html.Div([
					html.H2(subtitles['Step12.02']),
					dcc.Graph(id={'type':'graph', 'role': 'result', 'index': 'pred'}),
					], id={'type':'div', 'role': 'result', 'index': 'pred'}),
			html.Br(),
			html.Div([
					html.H2(subtitles['Step13']),
					html.Table(id={'type':'table', 'role': 'result', 'index': 'meas'}),
					], id={'type':'div', 'role': 'result', 'index': 'meas'}),
			html.Br(),
			html.Div([
					html.H2(subtitles['Step14']),#workout models display TODO
					html.Table(id={'type':'table', 'role': 'result', 'index': 'prop'}),
					], id={'type':'div', 'role': 'result', 'index': 'tmodel'}),
			], id={'type':'div', 'role': 'prompt', 'index': 'conf'})
		]);


# BEGIN CALLBACK FUNCTIONS
# BEGIN OPTION ACCESS CONTROL CALLBACKS
@app.callback(
    Output(component_id = {'type':'selector-uni', 'role':'choice', 'index': MATCH}, component_property = "disabled")
	,Input(component_id = {'type':'button', 'role': 'submit', 'index': MATCH}, component_property = 'n_clicks'))
def lockSelUni(n_clicks):
# disable changing a selection once the submit button has been hit
	return (n_clicks>0);
'''
@app.callback(
    Output(component_id = {'type':'selector-tree', 'role':'choice', 'index': MATCH}, component_property = "disabled")
	,State(component_id = {'type':'selector-tree', 'role':'choice', 'index': MATCH}, component_property = "value")
	,State(component_id = {'type':'selector-tree', 'role':'choice', 'index': MATCH}, component_property = "options")
	,Input(component_id = {'type':'button', 'role': 'submit', 'index': MATCH}, component_property = 'n_clicks'))
def lockSelMulti(predValue,predOptions,n_clicks):
# disable changing a selection once the submit button has been hit (more complicated for multi-select)
	newOptions = predOptions;
	if (n_clicks>0):
		return true;
	return newOptions;
'''
@app.callback(
    Output(component_id = {'type':'selector-multi', 'role':'choice', 'index': MATCH}, component_property = "options")
	,State(component_id = {'type':'selector-multi', 'role':'choice', 'index': MATCH}, component_property = "value")
	,State(component_id = {'type':'selector-multi', 'role':'choice', 'index': MATCH}, component_property = "options")
	,Input(component_id = {'type':'button', 'role': 'submit', 'index': MATCH}, component_property = 'n_clicks'))
def lockSelMulti(predValue,predOptions,n_clicks):
# disable changing a selection once the submit button has been hit (more complicated for multi-select)
	newOptions = predOptions;
	if (n_clicks>0):
		if (predValue is None):
			newOptions = [{"label": option["label"],"value": option["value"],"disabled": True} for option in newOptions];
		else: 
			newOptions = [{"label": option["label"],"value": option["value"]
						   ,"disabled": (option["value"] not in predValue)} 
						  for option in newOptions];
	return newOptions;
# END OPTION ACCESS CONTROL CALLBACKS
# BEGIN DISPLAY CONTROL CALLBACKS
@app.callback(Output(component_id = {'type':'div', 'role': 'prompt', 'index': MATCH}, component_property = 'style')
     ,Input(component_id = {'type':'button', 'role': 'submit', 'index': MATCH}, component_property = 'n_clicks')
)
def displayNextDiv(n_clicks):
# only display the next set of options once a user is ready to proceed
	newStyle = {'visibility':'hidden'};
	if (n_clicks>0):
		newStyle = {'visibility':'visible'};
	return newStyle;
	
@app.callback(Output(component_id = {'type':'button', 'role': 'submit', 'index': MATCH}, component_property = 'style')
     ,Input(component_id = {'type':'button', 'role': 'submit', 'index': MATCH}, component_property = 'n_clicks')
)
def hidePrevSubmit(n_clicks):
# remove each submit button as it is clicked to prevent flow irregularities
	newStyle = {'visibility':'inherited'};
	if (n_clicks>0):
		newStyle = {'visibility':'hidden'};
	return newStyle;

# BEGIN MODEL CONF DISPLAY CONTROL CALLBACKS
# pattern-matching on Output+State but not Input doesn't seem to be valid, replace this with multi-output later
@app.callback(Output(component_id = {'type':'div', 'role': 'conf', 'index': 'intercept'}, component_property = 'style')
				,State(component_id = {'type':'div', 'role': 'conf', 'index': 'intercept'}, component_property = 'id')
				,State(component_id = {'type':'selector-uni', 'role':'choice', 'index': 'prop'}, component_property = 'value')
				,Input(component_id = {'type':'button', 'role': 'submit', 'index': 'prop'}, component_property = 'n_clicks'))
def displayConfDiv(name,selProp,n_clicks):
	newStyle = {'visibility':'hidden'};
	if (n_clicks>0):
		propReqConf = checkConfReq(namesProp[selProp],name['index'].title());
		if (propReqConf):
			newStyle = {'visibility':'visible'};
	return newStyle;

@app.callback(Output(component_id = {'type':'div', 'role': 'conf', 'index': 'magnitude'}, component_property = 'style')
				,State(component_id = {'type':'div', 'role': 'conf', 'index': 'magnitude'}, component_property = 'id')
				,State(component_id = {'type':'selector-uni', 'role':'choice', 'index': 'prop'}, component_property = 'value')
				,Input(component_id = {'type':'button', 'role': 'submit', 'index': 'prop'}, component_property = 'n_clicks'))
def displayConfDiv(name,selProp,n_clicks):
	newStyle = {'visibility':'hidden'};
	if (n_clicks>0):
		propReqConf = checkConfReq(namesProp[selProp],name['index'].title());
		if (propReqConf):
			newStyle = {'visibility':'visible'};
	return newStyle;

@app.callback(Output(component_id = {'type':'div', 'role': 'conf', 'index': 'binthresh'}, component_property = 'style')
				,State(component_id = {'type':'div', 'role': 'conf', 'index': 'binthresh'}, component_property = 'id')
				,State(component_id = {'type':'selector-uni', 'role':'choice', 'index': 'prop'}, component_property = 'value')
				,Input(component_id = {'type':'button', 'role': 'submit', 'index': 'prop'}, component_property = 'n_clicks'))
def displayConfDiv(name,selProp,n_clicks):
	newStyle = {'visibility':'hidden'};
	if (n_clicks>0):
		propReqConf = checkConfReq(namesProp[selProp],name['index'].title());
		if (propReqConf):
			newStyle = {'visibility':'visible'};
	return newStyle;

@app.callback(Output(component_id = {'type':'div', 'role': 'conf', 'index': 'layers'}, component_property = 'style')
				,State(component_id = {'type':'div', 'role': 'conf', 'index': 'layers'}, component_property = 'id')
				,State(component_id = {'type':'selector-uni', 'role':'choice', 'index': 'prop'}, component_property = 'value')
				,Input(component_id = {'type':'button', 'role': 'submit', 'index': 'prop'}, component_property = 'n_clicks'))
def displayConfDiv(name,selProp,n_clicks):
	newStyle = {'visibility':'hidden'};
	if (n_clicks>0):
		propReqConf = checkConfReq(namesProp[selProp],name['index'].title());
		if (propReqConf):
			newStyle = {'visibility':'visible'};
	return newStyle;

@app.callback(Output(component_id = {'type':'div', 'role': 'conf', 'index': 'nodes'}, component_property = 'style')
				,State(component_id = {'type':'div', 'role': 'conf', 'index': 'nodes'}, component_property = 'id')
				,State(component_id = {'type':'selector-uni', 'role':'choice', 'index': 'prop'}, component_property = 'value')
				,Input(component_id = {'type':'button', 'role': 'submit', 'index': 'prop'}, component_property = 'n_clicks'))
def displayConfDiv(name,selProp,n_clicks):
	newStyle = {'visibility':'hidden'};
	if (n_clicks>0):
		propReqConf = checkConfReq(namesProp[selProp],name['index'].title());
		if (propReqConf):
			newStyle = {'visibility':'visible'};
	return newStyle;

@app.callback(Output(component_id = {'type':'div', 'role': 'conf', 'index': 'learnrate'}, component_property = 'style')
				,State(component_id = {'type':'div', 'role': 'conf', 'index': 'learnrate'}, component_property = 'id')
				,State(component_id = {'type':'selector-uni', 'role':'choice', 'index': 'prop'}, component_property = 'value')
				,Input(component_id = {'type':'button', 'role': 'submit', 'index': 'prop'}, component_property = 'n_clicks'))
def displayConfDiv(name,selProp,n_clicks):
	newStyle = {'visibility':'hidden'};
	if (n_clicks>0):
		propReqConf = checkConfReq(namesProp[selProp],name['index'].title());
		if (propReqConf):
			newStyle = {'visibility':'visible'};
	return newStyle;
# END MODEL CONF DISPLAY CONTROL CALLBACKS
# END DISPLAY CONTROL CALLBACKS

# BEGIN LOGIC CALLBACKS
@app.callback(
    Output(component_id = {'type':'graph', 'role': 'result', 'index': 'resp'}, component_property = 'figure')
    ,State(component_id = {'type':'selector-uni', 'role':'choice', 'index': 'resp'}, component_property = 'value')
    ,Input(component_id = {'type':'button', 'role': 'submit', 'index': 'conf'}, component_property = 'n_clicks'))
def genHistoRespCB(selResp, n_clicks):
# validate input -> pass to logic -> format logic result for display
	if (n_clicks>0):
		hist, bins = np.histogram(df_DataProc[namesResp[selResp]].to_numpy());
		retVal = {'data':[{'x':bins[0:-1],'y':hist,'type':'bar','name':namesResp[selResp]}]
				  ,'layout':{'title':"Response Variable Histogram"}};
	else:
		retVal = go.Figure(go.Bar(x=[], y = []))
		retVal.update_layout(template = None)
		retVal.update_xaxes(showgrid = False, showticklabels = False, zeroline=False)
		retVal.update_yaxes(showgrid = False, showticklabels = False, zeroline=False)
	return retVal; 

@app.callback(
    Output(component_id = {'type':'graph', 'role': 'result', 'index': 'pred'}, component_property = 'figure')
    ,State(component_id = {'type':'selector-tree', 'role':'choice', 'index': 'pred'}, component_property = 'checked')
    ,Input(component_id = {'type':'button', 'role': 'submit', 'index': 'conf'}, component_property = 'n_clicks'))
def genHistoPredCB(selPred, n_clicks):
# validate input -> pass to logic -> format logic result for display
	if (n_clicks>0):
		if (selPred is not None):
			selPred.sort();
			retVal = make_subplots(rows=1, cols=len(selPred)); colIdx=1;
			for pred in selPred:
				if (pred[0]!='#'):
					hist, bins = np.histogram(df_DataProc[pred].to_numpy());
					retVal.add_trace(go.Bar(x=bins[0:-1],y=hist,name=pred), row=1, col=colIdx)
					colIdx = colIdx+1;
	else:
		retVal = go.Figure(go.Bar(x=[], y = []))
		retVal.update_layout(template = None)
		retVal.update_xaxes(showgrid = False, showticklabels = False, zeroline=False)
		retVal.update_yaxes(showgrid = False, showticklabels = False, zeroline=False)
	return retVal;

@app.callback(
	Output(component_id = {'type':'table', 'role': 'result', 'index': 'meas'}, component_property = 'children')
    ,State(component_id = {'type':'selector-uni', 'role':'choice', 'index': 'resp'}, component_property = 'value')
    ,State(component_id = {'type':'selector-tree', 'role':'choice', 'index': 'pred'}, component_property = 'checked')
    ,State(component_id = {'type':'selector-tree', 'role':'choice', 'index': 'meas'}, component_property = 'checked')
    ,Input(component_id = {'type':'button', 'role': 'submit', 'index': 'conf'}, component_property = 'n_clicks'))
def genMeasCB(selResp,selPred,selMeas, n_clicks):
# validate input -> pass to logic -> format logic result for display
	if (n_clicks>0):
		retVal = '';
		if (selMeas is not None):
			selMeas.sort(); tableStyle = {'text-align':'center','border':'1px black solid'};
			selMeasUni = setInt(selMeas,list(uniList.keys()));
			if (selPred is not None):
				selPred.sort();
				measVal = genMeas(selResp, selPred, selMeas);
				header = [html.Th('Response Variable Measures',colSpan=max(len(selMeasUni),1),style=tableStyle)
							 ,html.Th('Predictor Variable Measures',colSpan=(len(selPred)*len(selMeas)),style=tableStyle)];
				subheader = [html.Td(namesResp[selResp],colSpan=len(selMeasUni),style=tableStyle)];
				for key in measVal['pred'].keys():
					subheader.append(html.Td(key,colSpan=len(selMeas),style=tableStyle));
				resultcols = [];
				if (len(selMeasUni)<1):
					resultcols.append(html.Td('',style=tableStyle));
				else:
					for respK in measVal['resp'].keys():
						resultcols.append(html.Td(respK,style=tableStyle));
				for predV in measVal['pred'].values():
					for key in predV:
						resultcols.append(html.Td(key,style=tableStyle));
				results = [];
				for respV in measVal['resp'].values():
					results.append(html.Td(respV,style=tableStyle));
				for predV in measVal['pred'].values():
					for key in predV:
						results.append(html.Td(predV[key],style=tableStyle));
			else:
				measVal = genMeas(selResp, [], selMeas);
				header = [html.Th('Response Variable Measures',colSpan=len(selMeasUni),style=tableStyle)];
				subheader = [html.Td(namesResp[selResp],colSpan=len(selMeasUni),style=tableStyle)];
				resultcols = [];
				for respK in measVal['resp'].keys():
					resultcols.append(html.Td(respK,style=tableStyle));
				results = [];
				for respV in measVal['resp'].values():
					results.append(html.Td(respV,style=tableStyle));
		retVal = html.Tbody([html.Tr(header),html.Tr(subheader),html.Tr(resultcols),html.Tr(results)]);
		return retVal;

@app.callback(Output(component_id = {'type':'table', 'role': 'result', 'index': 'prop'}, component_property = 'children')
    ,State(component_id = {'type':'selector-uni', 'role':'choice', 'index': 'resp'}, component_property = 'value')
    ,State(component_id = {'type':'selector-tree', 'role':'choice', 'index': 'pred'}, component_property = 'checked')
    ,State(component_id = {'type':'selector-uni', 'role':'choice', 'index': 'prop'}, component_property = 'value')
    ,State(component_id = {'type':'selector-uni', 'role':'conf', 'index': 'seed'}, component_property = 'value')
    ,State(component_id = {'type':'selector-uni', 'role':'conf', 'index': 'trainPct'}, component_property = 'value')
    ,State(component_id = {'type':'selector-uni', 'role':'conf', 'index': 'intercept'}, component_property = 'value')
    ,State(component_id = {'type':'selector-uni', 'role':'conf', 'index': 'magnitude'}, component_property = 'value')
    ,State(component_id = {'type':'selector-uni', 'role':'conf', 'index': 'binthresh'}, component_property = 'value')
    ,State(component_id = {'type':'selector-uni', 'role':'conf', 'index': 'layers'}, component_property = 'value')
    ,State(component_id = {'type':'selector-uni', 'role':'conf', 'index': 'nodes'}, component_property = 'value')
    ,State(component_id = {'type':'selector-uni', 'role':'conf', 'index': 'learnrate'}, component_property = 'value')			  
    ,Input(component_id = {'type':'button', 'role': 'submit', 'index': 'conf'}, component_property = 'n_clicks'))
def genModelCB(selResp, selPred, selProp, confSeed, confTPct, confInt, confMag, confBThr, confLyrs, confNodes, confLR, n_clicks):
# validate input -> transform data from user entry to logic entry -> pass to logic -> format logic result for display
	if (n_clicks>0):
		if (selPred is not None):
			selPred.sort(); d_Conf = {}; d_Conf['Seed'] = confSeed; 
			d_Conf['TrainPct'] = confTPct/100; d_Conf['Intercept'] = (confInt==1); 
			d_Conf['Magnitude'] = int(1*10**confMag); d_Conf['BinThresh'] = confBThr; 
			d_Conf['Layers'] = confLyrs; d_Conf['Nodes'] = confNodes; d_Conf['LearnRate'] = float(1*10**(-int(confLR))); 
			retVal = genModel(selResp, selPred, selProp, d_Conf);
			tableList = []; tableStyle = {'text-align':'center','border':'1px black solid'};
			propName = namesProp[selProp]; respName = namesResp[selResp]; binclass = checkConfReq(propName,'Binthresh');
			headLine = propName+" model of "+respName+" based on selected predictors:";
			rowHeader = html.Tr(html.Th(headLine,colSpan="100%",style=tableStyle));
			tableList.append(rowHeader);
			if (propName=='Neural-Network'):
				for layer,idx in zip(retVal['model'],range(len(retVal['model']))):
					if(not isinstance(layer,type(nn.ReLU()))):
						tableList.append(html.Tr([html.Td("Layer " + str(idx+1) + " weights:",style=tableStyle)
												  ,html.Td(layer.weight.detach().numpy().T,style=tableStyle)]));
			elif (propName=='Naive-Bayes-Gaussian'):
				tableList.append(html.Tr([html.Td("additive variance value",style=tableStyle)
										  ,html.Td("ε",style=tableStyle)
										  ,html.Td(retVal['model'].epsilon_[0],style=tableStyle)]));
				for pred,idx in zip(suppPred,range(len(suppPred))):
					tableList.append(html.Tr([html.Td(pred,style=tableStyle,colSpan="100%")]));
					tableList.append(html.Tr([html.Td(" ",style=tableStyle),html.Td("θ1±σ1",style=tableStyle)
											  ,html.Td(retVal['model'].theta_[idx][0]+"±"+retVal['model'].var_[idx][0],style=tableStyle)]));
					tableList.append(html.Tr([html.Td(" ",style=tableStyle),html.Td("θ2±σ2",style=tableStyle)
											  ,html.Td(retVal['model'].theta_[idx][1]+"±"+retVal['model'].var_[idx][1],style=tableStyle)]));
			else:
				suppPred = list(compress(selPred, retVal['features'].support_));
				if (propName=='Linear-Regression'):
					tableList.append(html.Tr([html.Td("(intercept)",style=tableStyle)
											  ,html.Td("β0",style=tableStyle)
											  ,html.Td(retVal['model'].intercept_,style=tableStyle)]));
					if (len(suppPred)>1):
						for coef,pred,idx in zip(retVal['model'].coef_,suppPred,range(len(retVal['model'].coef_))):
							tableList.append(html.Tr([html.Td(pred,style=tableStyle)
												  ,html.Td("β"+str(idx+1),style=tableStyle)
												  ,html.Td(coef,style=tableStyle)]));
					else:
						tableList.append(html.Tr([html.Td(suppPred[0],style=tableStyle),html.Td("β1",style=tableStyle),html.Td(retVal['model'].coef_[0],style=tableStyle)]));
				elif (propName=='Linear-Discriminant-Analysis' or propName=='Logistic-Regression' or propName=='Support-Vector-Machine'):
					tableList.append(html.Tr([html.Td("(intercept)",style=tableStyle)
											  ,html.Td("β0",style=tableStyle)
											  ,html.Td(retVal['model'].intercept_[0],style=tableStyle)]));
					if (len(suppPred)>1):
						for coef,pred,idx in zip(retVal['model'].coef_[0],suppPred,range(len(retVal['model'].coef_[0]))):
							tableList.append(html.Tr([html.Td(pred,style=tableStyle)
												  ,html.Td("β"+str(idx+1),style=tableStyle)
												  ,html.Td(coef,style=tableStyle)]));
					else:
						tableList.append(html.Tr([html.Td(suppPred[0],style=tableStyle),html.Td("β1",style=tableStyle),html.Td(retVal['model'].coef_[0][0],style=tableStyle)]));
			if ("error" in retVal.keys()):
				errorType = "%" if binclass else "RMS";
				tableList.append(html.Tr([html.Td("Error Rate on Training Data ("+errorType+"):",colSpan=2,style=tableStyle)
										  ,html.Td(retVal['error']['train'],style=tableStyle)]));
				tableList.append(html.Tr([html.Td("Error Rate on Test Data ("+errorType+"):",colSpan=2,style=tableStyle)
										  ,html.Td(retVal['error']['test'],style=tableStyle)]));
			if ("roc" in retVal.keys()):
				cellRocTrain = dcc.Graph(id='rocTrain',figure=makeROC(retVal['roc']['train'],'Training Data'));
				cellRocTest = dcc.Graph(id='rocTest',figure=makeROC(retVal['roc']['test'],'Test Data'));
				tableList.append(html.Tr([html.Td(cellRocTrain,colSpan="100%")]));
				tableList.append(html.Tr([html.Td(cellRocTest,colSpan="100%")]));
			retVal = html.Tbody(tableList);
		else:
			retVal = html.Tbody([]);
		return retVal;
# END LOGIC CALLBACKS
# END CALLBACK FUNCTIONS

def genCBTPred():
	retVal = [
		{"value":"#Burst","label":"Burst Measures","children": [
			{"value":"#Elec1B","label":"Burst Electrode 1","children":[{"label":v,"value":v} for k,v in genFieldDict(['Is_Burst','Is_Elec1'],df_Schema).items()]}
		   ,{"value":"#Elec2B","label":"Burst Electrode 2","children":[{"label":v,"value":v} for k,v in genFieldDict(['Is_Burst','Is_Elec2'],df_Schema).items()]}
		]}
	   ,{"value":"#ECG","label":"ECG and Derived Values","children": [
			{"value":"#Is_20Mins","label":"Measures at 20 Mins","children":[{"label":v,"value":v} for k,v in genFieldDict(['Is_ECG','Is_20Mins'],df_Schema).items()]}
		   ,{"value":"#Is_40Mins","label":"Measures at 40 Mins","children":[{"label":v,"value":v} for k,v in genFieldDict(['Is_ECG','Is_40Mins'],df_Schema).items()]}
		   ,{"value":"#Is_60Mins","label":"Measures at 60 Mins","children":[{"label":v,"value":v} for k,v in genFieldDict(['Is_ECG','Is_60Mins'],df_Schema).items()]}
		]}
		,{"value":"#EEG","label":"EEG and Derived Values","children": [
			{"value":"#Elec1","label":"Burst Electrode 1","children": [
				{"value":"#Elec1Power","label":"Spectral Power","children": [
						{"value":"#Elec1Power@20Mins","label":"Measures at 20 Mins","children": [
							{"value":"#Elec1Power@20MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','!Is_Frac','Is_20Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec1Power@20MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','!Is_Frac','Is_20Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec1Power@20MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','!Is_Frac','Is_20Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec1Power@20MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','!Is_Frac','Is_20Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec1Power@20MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','!Is_Frac','Is_20Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec1Power@20MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','!Is_Frac','Is_20Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec1Power@20MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','!Is_Frac','Is_20Mins','Is_Total'],df_Schema).items()]}
						]}
						,{"value":"#Elec1Power@40Mins","label":"Measures at 40 Mins","children": [
							{"value":"#Elec1Power@40MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','!Is_Frac','Is_40Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec1Power@40MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','!Is_Frac','Is_40Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec1Power@40MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','!Is_Frac','Is_40Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec1Power@40MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','!Is_Frac','Is_40Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec1Power@40MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','!Is_Frac','Is_40Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec1Power@40MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','!Is_Frac','Is_40Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec1Power@40MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','!Is_Frac','Is_40Mins','Is_Total'],df_Schema).items()]}
						]}
						,{"value":"#Elec1Power@60Mins","label":"Measures at 60 Mins","children":
						[
							{"value":"#Elec1Power@60MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','!Is_Frac','Is_60Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec1Power@60MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','!Is_Frac','Is_60Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec1Power@60MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','!Is_Frac','Is_60Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec1Power@60MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','!Is_Frac','Is_60Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec1Power@60MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','!Is_Frac','Is_60Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec1Power@60MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','!Is_Frac','Is_60Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec1Power@60MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','!Is_Frac','Is_60Mins','Is_Total'],df_Schema).items()]}
						]}
					]
				}
				,{"value":"#Elec1Shannon","label":"Shannon Entropy","children": [
						{"value":"#Elec1Shannon@20Mins","label":"Measures at 20 Mins","children": [
							{"value":"#Elec1Shannon@20MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','!Is_Frac','Is_20Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec1Shannon@20MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','!Is_Frac','Is_20Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec1Shannon@20MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','!Is_Frac','Is_20Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec1Shannon@20MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','!Is_Frac','Is_20Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec1Shannon@20MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','!Is_Frac','Is_20Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec1Shannon@20MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','!Is_Frac','Is_20Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec1Shannon@20MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','!Is_Frac','Is_20Mins','Is_Total'],df_Schema).items()]}
						]}
						,{"value":"#Elec1Shannon@40Mins","label":"Measures at 40 Mins","children": [
							{"value":"#Elec1Shannon@40MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','!Is_Frac','Is_40Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec1Shannon@40MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','!Is_Frac','Is_40Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec1Shannon@40MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','!Is_Frac','Is_40Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec1Shannon@40MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','!Is_Frac','Is_40Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec1Shannon@40MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','!Is_Frac','Is_40Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec1Shannon@40MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','!Is_Frac','Is_40Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec1Shannon@40MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','!Is_Frac','Is_40Mins','Is_Total'],df_Schema).items()]}
						]}
						,{"value":"#Elec1Shannon@60Mins","label":"Measures at 60 Mins","children": [
							{"value":"#Elec1Shannon@60MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','!Is_Frac','Is_60Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec1Shannon@60MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','!Is_Frac','Is_60Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec1Shannon@60MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','!Is_Frac','Is_60Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec1Shannon@60MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','!Is_Frac','Is_60Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec1Shannon@60MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','!Is_Frac','Is_60Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec1Shannon@60MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','!Is_Frac','Is_60Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec1Shannon@60MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','!Is_Frac','Is_60Mins','Is_Total'],df_Schema).items()]}
						]}
					]
				}
				,{"value":"#Elec1Tsallis","label":"Tsallis Entropy","children": [
						{"value":"#Elec1Tsallis@20Mins","label":"Measures at 20 Mins","children": [
							{"value":"#Elec1Tsallis@20MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','!Is_Frac','Is_20Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec1Tsallis@20MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','!Is_Frac','Is_20Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec1Tsallis@20MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','!Is_Frac','Is_20Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec1Tsallis@20MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','!Is_Frac','Is_20Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec1Tsallis@20MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','!Is_Frac','Is_20Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec1Tsallis@20MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','!Is_Frac','Is_20Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec1Tsallis@20MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','!Is_Frac','Is_20Mins','Is_Total'],df_Schema).items()]}
						]}
						,{"value":"#Elec1Tsallis@40Mins","label":"Measures at 40 Mins","children": [
							{"value":"#Elec1Tsallis@40MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','!Is_Frac','Is_40Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec1Tsallis@40MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','!Is_Frac','Is_40Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec1Tsallis@40MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','!Is_Frac','Is_40Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec1Tsallis@40MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','!Is_Frac','Is_40Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec1Tsallis@40MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','!Is_Frac','Is_40Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec1Tsallis@40MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','!Is_Frac','Is_40Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec1Tsallis@40MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','!Is_Frac','Is_40Mins','Is_Total'],df_Schema).items()]}
						]}
						,{"value":"#Elec1Tsallis@60Mins","label":"Measures at 60 Mins","children": [
							{"value":"#Elec1Tsallis@60MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','!Is_Frac','Is_60Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec1Tsallis@60MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','!Is_Frac','Is_60Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec1Tsallis@60MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','!Is_Frac','Is_60Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec1Tsallis@60MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','!Is_Frac','Is_60Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec1Tsallis@60MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','!Is_Frac','Is_60Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec1Tsallis@60MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','!Is_Frac','Is_60Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec1Tsallis@60MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','!Is_Frac','Is_60Mins','Is_Total'],df_Schema).items()]}
						]}
					]
				}
				,{"value":"#Elec1PowerFrac","label":"Fraction of Total Spectral Power","children": [
						{"value":"#Elec1PowerFrac@20Mins","label":"Measures at 20 Mins","children": [
							{"value":"#Elec1PowerFrac@20MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','Is_Frac','Is_20Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec1PowerFrac@20MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','Is_Frac','Is_20Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec1PowerFrac@20MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','Is_Frac','Is_20Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec1PowerFrac@20MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','Is_Frac','Is_20Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec1PowerFrac@20MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','Is_Frac','Is_20Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec1PowerFrac@20MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','Is_Frac','Is_20Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec1PowerFrac@20MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','Is_Frac','Is_20Mins','Is_Total'],df_Schema).items()]}
						]}
						,{"value":"#Elec1PowerFrac@40Mins","label":"Measures at 40 Mins","children": [
							{"value":"#Elec1PowerFrac@40MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','Is_Frac','Is_40Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec1PowerFrac@40MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','Is_Frac','Is_40Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec1PowerFrac@40MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','Is_Frac','Is_40Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec1PowerFrac@40MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','Is_Frac','Is_40Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec1PowerFrac@40MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','Is_Frac','Is_40Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec1PowerFrac@40MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','Is_Frac','Is_40Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec1PowerFrac@40MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','Is_Frac','Is_40Mins','Is_Total'],df_Schema).items()]}
						]}
						,{"value":"#Elec1PowerFrac@60Mins","label":"Measures at 60 Mins","children": [
							{"value":"#Elec1PowerFrac@60MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','Is_Frac','Is_60Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec1PowerFrac@60MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','Is_Frac','Is_60Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec1PowerFrac@60MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','Is_Frac','Is_60Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec1PowerFrac@60MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','Is_Frac','Is_60Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec1PowerFrac@60MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','Is_Frac','Is_60Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec1PowerFrac@60MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','Is_Frac','Is_60Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec1PowerFrac@60MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Power','Is_Frac','Is_60Mins','Is_Total'],df_Schema).items()]}
						]}
					]
				}
				,{"value":"#Elec1ShannonFrac","label":"Fraction of Shannon Entropy","children": [
						{"value":"#Elec1ShannonFrac@20Mins","label":"Measures at 20 Mins","children": [
							{"value":"#Elec1ShannonFrac@20MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','Is_Frac','Is_20Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec1ShannonFrac@20MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','Is_Frac','Is_20Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec1ShannonFrac@20MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','Is_Frac','Is_20Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec1ShannonFrac@20MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','Is_Frac','Is_20Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec1ShannonFrac@20MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','Is_Frac','Is_20Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec1ShannonFrac@20MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','Is_Frac','Is_20Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec1ShannonFrac@20MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','Is_Frac','Is_20Mins','Is_Total'],df_Schema).items()]}
						]}
						,{"value":"#Elec1ShannonFrac@40Mins","label":"Measures at 40 Mins","children": [
							{"value":"#Elec1ShannonFrac@40MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','Is_Frac','Is_40Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec1ShannonFrac@40MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','Is_Frac','Is_40Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec1ShannonFrac@40MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','Is_Frac','Is_40Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec1ShannonFrac@40MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','Is_Frac','Is_40Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec1ShannonFrac@40MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','Is_Frac','Is_40Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec1ShannonFrac@40MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','Is_Frac','Is_40Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec1ShannonFrac@40MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','Is_Frac','Is_40Mins','Is_Total'],df_Schema).items()]}
						]}
						,{"value":"#Elec1ShannonFrac@60Mins","label":"Measures at 60 Mins","children": [
							{"value":"#Elec1ShannonFrac@60MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','Is_Frac','Is_60Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec1ShannonFrac@60MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','Is_Frac','Is_60Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec1ShannonFrac@60MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','Is_Frac','Is_60Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec1ShannonFrac@60MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','Is_Frac','Is_60Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec1ShannonFrac@60MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','Is_Frac','Is_60Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec1ShannonFrac@60MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','Is_Frac','Is_60Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec1ShannonFrac@60MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Shannon','Is_Frac','Is_60Mins','Is_Total'],df_Schema).items()]}
						]}
					]
				}
				,{"value":"#Elec1TsallisFrac","label":"Fraction of Tsallis Entropy","children": [
						{"value":"#Elec1TsallisFrac@20Mins","label":"Measures at 20 Mins","children": [
							{"value":"#Elec1TsallisFrac@20MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','Is_Frac','Is_20Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec1TsallisFrac@20MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','Is_Frac','Is_20Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec1TsallisFrac@20MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','Is_Frac','Is_20Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec1TsallisFrac@20MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','Is_Frac','Is_20Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec1TsallisFrac@20MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','Is_Frac','Is_20Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec1TsallisFrac@20MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','Is_Frac','Is_20Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec1TsallisFrac@20MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','Is_Frac','Is_20Mins','Is_Total'],df_Schema).items()]}
						]}
						,{"value":"#Elec1TsallisFrac@40Mins","label":"Measures at 40 Mins","children":
						[
							{"value":"#Elec1TsallisFrac@40MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','Is_Frac','Is_40Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec1TsallisFrac@40MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','Is_Frac','Is_40Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec1TsallisFrac@40MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','Is_Frac','Is_40Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec1TsallisFrac@40MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','Is_Frac','Is_40Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec1TsallisFrac@40MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','Is_Frac','Is_40Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec1TsallisFrac@40MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','Is_Frac','Is_40Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec1TsallisFrac@40MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','Is_Frac','Is_40Mins','Is_Total'],df_Schema).items()]}
						]}
						,{"value":"#Elec1TsallisFrac@60Mins","label":"Measures at 60 Mins","children":
						[
							{"value":"#Elec1TsallisFrac@60MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','Is_Frac','Is_60Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec1TsallisFrac@60MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','Is_Frac','Is_60Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec1TsallisFrac@60MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','Is_Frac','Is_60Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec1TsallisFrac@60MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','Is_Frac','Is_60Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec1TsallisFrac@60MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','Is_Frac','Is_60Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec1TsallisFrac@60MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','Is_Frac','Is_60Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec1TsallisFrac@60MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec1','Is_Entropy','Is_Tsallis','Is_Frac','Is_60Mins','Is_Total'],df_Schema).items()]}
						]}
					]
				}
			]
		}
		,{"value":"#Elec2","label":"Burst Electrode 2","children": [
				{"value":"#Elec2Power","label":"Spectral Power","children": [
						{"value":"#Elec2Power@20Mins","label":"Measures at 20 Mins","children": [
							{"value":"#Elec2Power@20MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','!Is_Frac','Is_20Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec2Power@20MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','!Is_Frac','Is_20Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec2Power@20MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','!Is_Frac','Is_20Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec2Power@20MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','!Is_Frac','Is_20Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec2Power@20MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','!Is_Frac','Is_20Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec2Power@20MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','!Is_Frac','Is_20Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec2Power@20MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','!Is_Frac','Is_20Mins','Is_Total'],df_Schema).items()]}
						]}
						,{"value":"#Elec2Power@40Mins","label":"Measures at 40 Mins","children": [
							{"value":"#Elec2Power@40MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','!Is_Frac','Is_40Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec2Power@40MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','!Is_Frac','Is_40Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec2Power@40MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','!Is_Frac','Is_40Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec2Power@40MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','!Is_Frac','Is_40Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec2Power@40MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','!Is_Frac','Is_40Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec2Power@40MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','!Is_Frac','Is_40Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec2Power@40MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','!Is_Frac','Is_40Mins','Is_Total'],df_Schema).items()]}
						]}
						,{"value":"#Elec2Power@60Mins","label":"Measures at 60 Mins","children": [
							{"value":"#Elec2Power@60MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','!Is_Frac','Is_60Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec2Power@60MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','!Is_Frac','Is_60Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec2Power@60MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','!Is_Frac','Is_60Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec2Power@60MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','!Is_Frac','Is_60Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec2Power@60MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','!Is_Frac','Is_60Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec2Power@60MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','!Is_Frac','Is_60Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec2Power@60MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','!Is_Frac','Is_60Mins','Is_Total'],df_Schema).items()]}
						]}
					]
				}
				,{"value":"#Elec2Shannon","label":"Shannon Entropy","children": [
						{"value":"#Elec2Shannon@20Mins","label":"Measures at 20 Mins","children": [
							{"value":"#Elec2Shannon@20MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','!Is_Frac','Is_20Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec2Shannon@20MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','!Is_Frac','Is_20Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec2Shannon@20MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','!Is_Frac','Is_20Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec2Shannon@20MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','!Is_Frac','Is_20Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec2Shannon@20MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','!Is_Frac','Is_20Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec2Shannon@20MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','!Is_Frac','Is_20Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec2Shannon@20MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','!Is_Frac','Is_20Mins','Is_Total'],df_Schema).items()]}
						]}
						,{"value":"#Elec2Shannon@40Mins","label":"Measures at 40 Mins","children":
						[
							{"value":"#Elec2Shannon@40MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','!Is_Frac','Is_40Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec2Shannon@40MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','!Is_Frac','Is_40Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec2Shannon@40MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','!Is_Frac','Is_40Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec2Shannon@40MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','!Is_Frac','Is_40Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec2Shannon@40MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','!Is_Frac','Is_40Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec2Shannon@40MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','!Is_Frac','Is_40Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec2Shannon@40MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','!Is_Frac','Is_40Mins','Is_Total'],df_Schema).items()]}
						]}
						,{"value":"#Elec2Shannon@60Mins","label":"Measures at 60 Mins","children": [
							{"value":"#Elec2Shannon@60MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','!Is_Frac','Is_60Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec2Shannon@60MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','!Is_Frac','Is_60Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec2Shannon@60MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','!Is_Frac','Is_60Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec2Shannon@60MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','!Is_Frac','Is_60Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec2Shannon@60MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','!Is_Frac','Is_60Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec2Shannon@60MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','!Is_Frac','Is_60Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec2Shannon@60MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','!Is_Frac','Is_60Mins','Is_Total'],df_Schema).items()]}
						]}
					]
				}
				,{"value":"#Elec2Tsallis","label":"Tsallis Entropy","children": [
						{"value":"#Elec2Tsallis@20Mins","label":"Measures at 20 Mins","children": [
							{"value":"#Elec2Tsallis@20MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','!Is_Frac','Is_20Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec2Tsallis@20MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','!Is_Frac','Is_20Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec2Tsallis@20MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','!Is_Frac','Is_20Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec2Tsallis@20MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','!Is_Frac','Is_20Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec2Tsallis@20MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','!Is_Frac','Is_20Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec2Tsallis@20MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','!Is_Frac','Is_20Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec2Tsallis@20MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','!Is_Frac','Is_20Mins','Is_Total'],df_Schema).items()]}
						]}
						,{"value":"#Elec2Tsallis@40Mins","label":"Measures at 40 Mins","children": [
							{"value":"#Elec2Tsallis@40MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','!Is_Frac','Is_40Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec2Tsallis@40MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','!Is_Frac','Is_40Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec2Tsallis@40MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','!Is_Frac','Is_40Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec2Tsallis@40MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','!Is_Frac','Is_40Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec2Tsallis@40MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','!Is_Frac','Is_40Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec2Tsallis@40MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','!Is_Frac','Is_40Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec2Tsallis@40MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','!Is_Frac','Is_40Mins','Is_Total'],df_Schema).items()]}
						]}
						,{"value":"#Elec2Tsallis@60Mins","label":"Measures at 60 Mins","children": [
							{"value":"#Elec2Tsallis@60MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','!Is_Frac','Is_60Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec2Tsallis@60MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','!Is_Frac','Is_60Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec2Tsallis@60MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','!Is_Frac','Is_60Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec2Tsallis@60MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','!Is_Frac','Is_60Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec2Tsallis@60MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','!Is_Frac','Is_60Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec2Tsallis@60MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','!Is_Frac','Is_60Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec2Tsallis@60MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','!Is_Frac','Is_60Mins','Is_Total'],df_Schema).items()]}
						]}
					]
				}
				,{"value":"#Elec2PowerFrac","label":"Fraction of Total Spectral Power","children": [
						{"value":"#Elec2PowerFrac@20Mins","label":"Measures at 20 Mins","children": [
							{"value":"#Elec2PowerFrac@20MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','Is_Frac','Is_20Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec2PowerFrac@20MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','Is_Frac','Is_20Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec2PowerFrac@20MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','Is_Frac','Is_20Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec2PowerFrac@20MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','Is_Frac','Is_20Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec2PowerFrac@20MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','Is_Frac','Is_20Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec2PowerFrac@20MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','Is_Frac','Is_20Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec2PowerFrac@20MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','Is_Frac','Is_20Mins','Is_Total'],df_Schema).items()]}
						]}
						,{"value":"#Elec2PowerFrac@40Mins","label":"Measures at 40 Mins","children":
						[
							{"value":"#Elec2PowerFrac@40MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','Is_Frac','Is_40Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec2PowerFrac@40MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','Is_Frac','Is_40Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec2PowerFrac@40MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','Is_Frac','Is_40Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec2PowerFrac@40MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','Is_Frac','Is_40Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec2PowerFrac@40MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','Is_Frac','Is_40Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec2PowerFrac@40MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','Is_Frac','Is_40Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec2PowerFrac@40MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','Is_Frac','Is_40Mins','Is_Total'],df_Schema).items()]}
						]}
						,{"value":"#Elec2PowerFrac@60Mins","label":"Measures at 60 Mins","children": [
							{"value":"#Elec2PowerFrac@60MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','Is_Frac','Is_60Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec2PowerFrac@60MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','Is_Frac','Is_60Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec2PowerFrac@60MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','Is_Frac','Is_60Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec2PowerFrac@60MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','Is_Frac','Is_60Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec2PowerFrac@60MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','Is_Frac','Is_60Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec2PowerFrac@60MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','Is_Frac','Is_60Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec2PowerFrac@60MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Power','Is_Frac','Is_60Mins','Is_Total'],df_Schema).items()]}
						]}
					]
				}
				,{"value":"#Elec2ShannonFrac","label":"Fraction of Shannon Entropy","children": [
						{"value":"#Elec2ShannonFrac@20Mins","label":"Measures at 20 Mins","children": [
							{"value":"#Elec2ShannonFrac@20MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','Is_Frac','Is_20Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec2ShannonFrac@20MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','Is_Frac','Is_20Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec2ShannonFrac@20MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','Is_Frac','Is_20Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec2ShannonFrac@20MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','Is_Frac','Is_20Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec2ShannonFrac@20MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','Is_Frac','Is_20Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec2ShannonFrac@20MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','Is_Frac','Is_20Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec2ShannonFrac@20MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','Is_Frac','Is_20Mins','Is_Total'],df_Schema).items()]}
						]}
						,{"value":"#Elec2ShannonFrac@40Mins","label":"Measures at 40 Mins","children":
						[
							{"value":"#Elec2ShannonFrac@40MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','Is_Frac','Is_40Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec2ShannonFrac@40MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','Is_Frac','Is_40Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec2ShannonFrac@40MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','Is_Frac','Is_40Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec2ShannonFrac@40MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','Is_Frac','Is_40Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec2ShannonFrac@40MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','Is_Frac','Is_40Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec2ShannonFrac@40MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','Is_Frac','Is_40Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec2ShannonFrac@40MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','Is_Frac','Is_40Mins','Is_Total'],df_Schema).items()]}
						]}
						,{"value":"#Elec2ShannonFrac@60Mins","label":"Measures at 60 Mins","children":
						[
							{"value":"#Elec2ShannonFrac@60MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','Is_Frac','Is_60Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec2ShannonFrac@60MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','Is_Frac','Is_60Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec2ShannonFrac@60MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','Is_Frac','Is_60Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec2ShannonFrac@60MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','Is_Frac','Is_60Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec2ShannonFrac@60MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','Is_Frac','Is_60Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec2ShannonFrac@60MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','Is_Frac','Is_60Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec2ShannonFrac@60MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Shannon','Is_Frac','Is_60Mins','Is_Total'],df_Schema).items()]}
						]}
					]
				}
				,{"value":"#Elec2TsallisFrac","label":"Fraction of Tsallis Entropy","children":
					[
						{"value":"#Elec2TsallisFrac@20Mins","label":"Measures at 20 Mins","children":
						[
							{"value":"#Elec2TsallisFrac@20MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','Is_Frac','Is_20Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec2TsallisFrac@20MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','Is_Frac','Is_20Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec2TsallisFrac@20MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','Is_Frac','Is_20Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec2TsallisFrac@20MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','Is_Frac','Is_20Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec2TsallisFrac@20MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','Is_Frac','Is_20Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec2TsallisFrac@20MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','Is_Frac','Is_20Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec2TsallisFrac@20MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','Is_Frac','Is_20Mins','Is_Total'],df_Schema).items()]}
						]}
						,{"value":"#Elec2TsallisFrac@40Mins","label":"Measures at 40 Mins","children":
						[
							{"value":"#Elec2TsallisFrac@40MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','Is_Frac','Is_40Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec2TsallisFrac@40MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','Is_Frac','Is_40Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec2TsallisFrac@40MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','Is_Frac','Is_40Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec2TsallisFrac@40MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','Is_Frac','Is_40Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec2TsallisFrac@40MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','Is_Frac','Is_40Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec2TsallisFrac@40MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','Is_Frac','Is_40Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec2TsallisFrac@40MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','Is_Frac','Is_40Mins','Is_Total'],df_Schema).items()]}
						]}
						,{"value":"#Elec2TsallisFrac@60Mins","label":"Measures at 60 Mins","children":
						[
							{"value":"#Elec2TsallisFrac@60MinsDelta","label":"Delta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','Is_Frac','Is_60Mins','Is_Delta'],df_Schema).items()]}
							,{"value":"#Elec2TsallisFrac@60MinsTheta","label":"Theta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','Is_Frac','Is_60Mins','Is_Theta'],df_Schema).items()]}
							,{"value":"#Elec2TsallisFrac@60MinsAlpha","label":"Alpha Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','Is_Frac','Is_60Mins','Is_Alpha'],df_Schema).items()]}
							,{"value":"#Elec2TsallisFrac@60MinsBeta","label":"Beta Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','Is_Frac','Is_60Mins','Is_Beta'],df_Schema).items()]}
							,{"value":"#Elec2TsallisFrac@60MinsGamma","label":"Gamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','Is_Frac','Is_60Mins','Is_Gamma'],df_Schema).items()]}
							,{"value":"#Elec2TsallisFrac@60MinsSuperGamma","label":"SuperGamma Band","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','Is_Frac','Is_60Mins','Is_SuperGamma'],df_Schema).items()]}
							,{"value":"#Elec2TsallisFrac@60MinsTotal","label":"Total","children":[
								{"label":v,"value":v} for k,v in genFieldDict(['Is_EEG','Is_Elec2','Is_Entropy','Is_Tsallis','Is_Frac','Is_60Mins','Is_Total'],df_Schema).items()]}
						]}
					]
				}
			]
		}
	]}];
	return retVal;

def genCBTMeas():
	retVal = [
		{"value":"#Basic","label":"Basic Measures","children": [
			{"label":v,"value":v} for k,v in genFieldDict(['Basic','Ready'],df_Meas).items()
		]}
		,{"value":"#Advanced","label":"Advanced and Derivative Measures","children": [
			{"label":v,"value":v} for k,v in genFieldDict(['Advanced','Ready'],df_Meas).items()
		]}
		,{"value":"#Complex","label":"Complex Relationship Measures","children": [
			{"label":v,"value":v} for k,v in genFieldDict(['Complex','Ready'],df_Meas).items()
		]}
	];
	return retVal;

# BEGIN LOGIC FUNCTIONS
def loadData():
#. load data files into memory
    global df_Cohorts; df_Cohorts = pd.read_csv(folder+'/'+files[0]); 
    global df_DataProc; df_DataProc = pd.read_csv(folder+'/'+files[1]);
    global df_Meas; df_Meas = pd.read_csv(folder+'/'+files[2]); 
    global df_Prop; df_Prop = pd.read_csv(folder+'/'+files[3]);
    global df_Schema; df_Schema = pd.read_csv(folder+'/'+files[4]);
    global namesResp; namesResp = genFieldDict(['Response'],df_Schema); 
    global namesPred; namesPred = genFieldDict(['Predictor'],df_Schema);
    global namesMeas; namesMeas = genFieldDict(['Ready'],df_Meas); 
    global namesProp; namesProp = genFieldDict(['Ready'],df_Prop);
    global uniList; uniList = genFieldDict(['Variate'],df_Meas);

def genFieldDict(reqList,df):
#. generate field dictionary from dataframe based on specified parameter flag(s)
    reqQuery = buildQuery(reqList);
    fieldlist = df.query(reqQuery).Column.to_numpy();
    fieldDict = dict(enumerate(fieldlist.flatten(), 1));
    #fieldDictInv = dict((v, k) for k, v in fieldDictInv.items());
    return fieldDict;
    
def buildQuery(colList,valList = [1], ander=True):
    conj = " and " if ander else " or ";
    offset = len(conj); query = ""; eqstr = " == ";
    if (len(colList)>1):
        if (len(valList)>1):
            queryDict = dict(zip(colList,valList));
        else:
            queryDict = dict(zip(colList,valList*len(colList)));
        for k,v in queryDict.items():
            if (k[0]=='!'):
                query = query + conj + k[1:] + eqstr + str(int(not v));
            else:
                query = query + conj + k + eqstr + str(v);
        query = query[offset:(len(query))];
    else:
        query = str(colList[0]) + eqstr + str(valList[0]);
    return query;
    
def genMeas(varResp, varPred, varMeas):
# pull specified values from data sources -> pass to measure switch -> pass back to view
	retResp = {}; retVal = {}; measList = []; 
	for meas in varMeas:
		measList.append(meas);
	respVals = df_DataProc[namesResp[varResp]];     
	for meas in measList:
		if (meas in uniList.values()):
			retResp[meas] = calcSwitch(meas,respVals);
	if (len(varPred)>0):
		predDict = {};
		for varP in varPred:
			if (varP[0]!='#'):
				predVals = df_DataProc[varP];
				retPredCurr = {}; 
				for meas in measList:
					if (meas in uniList.values()):
						retPredCurr[meas] = calcSwitch(meas,predVals);
					else:
						retPredCurr[meas] = calcSwitch(meas,predVals,respVals);            
				predDict[varP] = retPredCurr;
				retVal['pred'] = predDict;
	retVal['resp'] = retResp;
	return retVal;

def calcSwitch(measName,varA,varB=[]):
# identify measure -> pass values to measure-specific function -> format result as single string -> pass back to view
    retVal = '';
# Bivariate Measures
    if (measName=='Chi-Squared-Independence'):
        val,pval = calcChiSq(varA,varB);        
        retVal = str(val)+"("+str(pval)+")";
    elif (measName=='Correlation-Pearson'):
        val,pval = calcCorrP(varA,varB);        
        retVal = str(val)+"("+str(pval)+")";
    elif (measName=='Correlation-Spearman'):
        val,pval = calcCorrS(varA,varB);
        retVal = str(val)+"("+str(pval)+")";
    elif (measName=='Covariance'):
        retVal = calcCovar(varA,varB);
# Univariate Measures
    elif (measName=='Interquartile-Range'):
        retVal = calcIQR(varA);
    elif (measName=='Kurtosis'):
        retVal = calcKurtosis(varA);
    elif (measName=='Maximum'):
        retVal = calcMax(varA);
    elif (measName=='Mean'):
        retVal = calcMean(varA);
    elif (measName=='Median'):
        retVal = calcMedian(varA);
    elif (measName=='Median-Absolute-Deviation'):
        retVal = calcMAD(varA);
    elif (measName=='Minimum'):
        retVal = calcMin(varA);
    elif (measName=='Mode'):
        retVal = calcMode(varA);
    elif (measName=='Normality'):
        val,pval = calcNormality(varA,varB);
        retVal = str(val)+"("+str(pval)+")";
    elif (measName=='Range'):
        retVal = calcRange(varA);
    elif (measName=='Relative-Standard-Deviation'):
        retVal = calcRSD(varA);
    elif (measName=='Skew'):
        retVal = calcSkew(varA);
    elif (measName=='Standard-Deviation'):
        retVal = calcStd(varA);
    return retVal;

# BEGIN MEASURE-SPECIFIC FUNCTIONS
# BEGIN BIVARIATE MEASURES
def calcChiSq(varA,varB):
    retV, retP, _, _ = chi2_contingency(np.array([varA,varB]), correction=False);
    retV = round(retV,MRL);
    retP = round(retP,MRL);
    return [retV,retP];

def calcCorrP(varA,varB):
    retV, retP = pearsonr(varA,varB); 
    retV = round(retV,MRL);
    retP = round(retP,MRL);
    return [retV,retP];

def calcCorrS(varA,varB):
    retV, retP = spearmanr(varA,varB); 
    retV = round(retV,MRL);
    retP = round(retP,MRL);
    return [retV,retP];

def calcCovar(varA,varB):
    return round(np.cov(varA,varB)[0][1],MRL);

def calcNormality(varA,varB): #TODO
    retV, retP = spearmanr(varList); 
    retV = round(retV,MRL);
    retP = round(retP,MRL);
    return [retV,retP];
# END BIVARIATE MEASURES
# BEGIN UNIVARIATE MEASURES
def calcIQR(varList):
    return round(iqr(varList),MRL);

def calcKurtosis(varList):
    return round(kurtosis(varList),MRL);

def calcMax(varList):
    return round(max(varList),MRL);

def calcMedian(varList):
    return round(stttx.median(varList),MRL);

def calcMAD(varList):
    return round(median_abs_deviation(varList),MRL);

def calcMean(varList):
    return round(np.mean(varList),MRL);

def calcMin(varList):
    return round(min(varList),MRL);

def calcMode(varList): 
    return mode(varList);

def calcRange(varList):
    return round(calcMax(varList)-calcMin(varList),MRL);

def calcRSD(varList):
    return round(calcStd(varList)/calcMean(varList),MRL);

def calcSkew(varList):
    return round(skew(varList),MRL);

def calcStd(varList):
    return round(np.std(varList),MRL);
# END UNIVARIATE MEASURES
# END MEASURE-SPECIFIC FUNCTIONS

def genModel(varResp, varPred, varProp, d_Conf):
# pull specified values from data sources -> pass to model switch -> pass back to view
	retVal = {}; 
	respName = namesResp[varResp]; valsResp = df_DataProc[respName];
	valsPred = pd.DataFrame(); 
	for varP in varPred:
		if (varP[0]!='#'):
			valsPred[varP] = df_DataProc[varP];
	propName = namesProp[varProp];
	retVal = modelSwitch(propName,valsResp,valsPred,d_Conf);
	return retVal;

def modelSwitch(propName,valsResp,valsPred,d_Conf):
# identify proposition -> pass values and configuration to proposition-specific function -> calculate error rates -> pass back to view
# TODO Display: Scatter plot(s) for Proposition:Linear Regression, Proposition:LDA, Proposition:SVM
# Node-Graph for Proposition:Decision Tree and Proposition:Neural Network
# Univariate: Box-and-whisker plot
# Bivariate: Scatter Plots
	retVal = {};
	np.random.seed(d_Conf['Seed']); sample = np.random.uniform(size = len(valsResp.index)) < d_Conf['TrainPct'];
	trainResp = valsResp[sample]; testResp = valsResp[~sample];
	trainPred = valsPred[sample]; testPred = valsPred[~sample];
	binclass = checkConfReq(propName,'Binthresh');
	if(binclass):
		thresh = np.percentile(valsResp,d_Conf['BinThresh']);
		trainResp = (trainResp > thresh).astype(int); 
		testResp = (testResp > thresh).astype(int); 
	if (propName=='Neural-Network'):
		model = modelNN(trainResp,trainPred,d_Conf);
		retVal['model'] = model;
		modTrain = model(convNNType(trainPred)).detach().numpy().T[0];
		modTest =  model(convNNType(testPred)).detach().numpy().T[0];
	elif (propName=='Decision-Tree'):
		retVal['model'] = modelDecTree(trainResp,trainPred,d_Conf);
		modTrain = retVal['model'].predict(trainPred);
		modTest = retVal['model'].predict(testPred);
	else:
		if (propName=='Linear-Discriminant-Analysis'):
			retVal['model'],retVal['features'] = modelLDA(trainResp,trainPred,d_Conf);
		elif (propName=='Linear-Regression'):
			retVal['model'],retVal['features'] = modelLinReg(trainResp,trainPred,d_Conf);
		elif (propName=='Logistic-Regression'):
			retVal['model'],retVal['features'] = modelLogReg(trainResp,trainPred,d_Conf);
		elif (propName=='Naive-Bayes-Categorical'):
			retVal['model'],retVal['features'] = modelNBCat(trainResp,trainPred,d_Conf);
		elif (propName=='Naive-Bayes-Gaussian'):
			retVal['model'],retVal['features'] = modelNBGauss(trainResp,trainPred,d_Conf);
		elif (propName=='Support-Vector-Machine'):
			retVal['model'],retVal['features'] = modelSVM(trainResp,trainPred,d_Conf);
		trainPredTF = retVal['features'].transform(trainPred);
		testPredTF = retVal['features'].transform(testPred);
		modTrain = retVal['model'].predict(trainPredTF);
		modTest = retVal['model'].predict(testPredTF);
	retVal['error'] = {};
	retVal['error']['train'] = assessErr(trainResp.to_numpy(),modTrain,binclass);
	retVal['error']['test'] = assessErr(testResp.to_numpy(),modTest,binclass);
	if (binclass):
		retVal['roc'] = {};
		retVal['roc']['train'] = assessAUC(trainResp.to_numpy(),modTrain);
		retVal['roc']['test'] = assessAUC(testResp.to_numpy(),modTest);
	return retVal;

# BEGIN PROPOSITION-SPECIFIC FUNCTIONS
def modelDecTree(resp,pred,d_Conf):
	model = tree.DecisionTreeClassifier(); 
	model.fit(pred,resp);
	return model;

def modelLDA(resp,pred,d_Conf):
	estimator = lda.LinearDiscriminantAnalysis(tol=d_Conf['LearnRate']); 
	estimator.fit(pred,resp);
	featureSelect = RFECV(estimator,min_features_to_select=2,cv=2).fit(pred,resp);
	predFS = featureSelect.transform(pred);
	fMask = featureSelect.support_;
	model = lda.LinearDiscriminantAnalysis(tol=d_Conf['LearnRate']); 
	model.fit(predFS,resp);
	return model,featureSelect;

def modelLinReg(resp,pred,d_Conf):
	estimator = lm.LinearRegression(fit_intercept=d_Conf['Intercept']);
	estimator.fit(pred,resp);
	featureSelect = RFECV(estimator,min_features_to_select=2,cv=2).fit(pred,resp);
	predFS = featureSelect.transform(pred);
	fMask = featureSelect.support_;
	model = lm.LinearRegression(fit_intercept=d_Conf['Intercept']);
	model.fit(predFS,resp);
	return model,featureSelect;

def modelLogReg(resp,pred,d_Conf):
	estimator = lm.LogisticRegression(fit_intercept=d_Conf['Intercept'],max_iter=d_Conf['Magnitude']);
	estimator.fit(pred,resp);
	featureSelect = RFECV(estimator,min_features_to_select=2,cv=2).fit(pred,resp);
	predFS = featureSelect.transform(pred);
	fMask = featureSelect.support_;
	model = lm.LogisticRegression(fit_intercept=d_Conf['Intercept'],max_iter=d_Conf['Magnitude']);
	model.fit(predFS,resp);    
	return model,featureSelect;

def modelNBCat(resp,pred,d_Conf):
	estimator = nb.CategoricalNB();
	estimator.fit(pred,resp);
	featureSelect = RFECV(estimator,min_features_to_select=2,cv=2).fit(pred,resp);
	predFS = featureSelect.transform(pred);
	fMask = featureSelect.support_;
	model = nb.CategoricalNB();
	model.fit(predFS,resp);    
	return model,featureSelect;

def modelNBGauss(resp,pred,d_Conf):
	estimator = nb.GaussianNB();
	estimator.fit(pred,resp);
	featureSelect = RFECV(estimator,min_features_to_select=2,cv=2).fit(pred,resp);
	predFS = featureSelect.transform(pred);
	fMask = featureSelect.support_;
	model = nb.GaussianNB();
	model.fit(predFS,resp);    
	return model,featureSelect;

def modelSVM(resp,pred,d_Conf):
	estimator = svm.SVC(kernel='linear',max_iter=d_Conf['Magnitude'],tol=d_Conf['LearnRate']);
	estimator.fit(pred,resp);
	featureSelect = RFECV(estimator,min_features_to_select=2,cv=2).fit(pred,resp);
	predFS = featureSelect.transform(pred);
	model = svm.SVC(kernel='linear',max_iter=d_Conf['Magnitude'],tol=d_Conf['LearnRate']);
	model.fit(predFS,resp);    
	return model,featureSelect;

def convNNType(df):
    return torch.from_numpy(df.values).float();

def modelNN(rdat,pdat,d_Conf):
    pdat = convNNType(pdat); rdat = convNNType(rdat);
    inSz = pdat.shape[1]; outSz = 1;
    modelGraph = OrderedDict([('inLayer', nn.Linear(inSz,d_Conf['Nodes'])),('relu1', nn.ReLU())]);
    if (d_Conf['Layers']>1):
        for idx in range(1,d_Conf['Layers']):
            modelGraph['hl'+str(idx)] = nn.Linear(d_Conf['Nodes'],d_Conf['Nodes']); 
            modelGraph['relu'+str(1+idx)] = nn.ReLU();
    modelGraph['outLayer'] = nn.Linear(d_Conf['Nodes'],outSz); model = nn.Sequential(modelGraph);    
    model.zero_grad(); lossFn = nn.MSELoss(reduction='sum');
    optim = torch.optim.Adam(model.parameters(), d_Conf['LearnRate']);
    for idx in range(d_Conf['Magnitude']):
        currPred = model(pdat); currLoss = lossFn(currPred,rdat);
        optim.zero_grad(); currLoss.backward(); optim.step();
    return model; 
# END PROPOSITION-SPECIFIC FUNCTIONS

def assessErr(truth,prediction,bindata=False):
    if (bindata): # Percent Error
        retVal = sum(abs(np.subtract(truth,prediction))/len(truth));
    else: # RMS Error
        retVal = round(math.sqrt(sum((np.subtract(truth,prediction))**2)/len(truth)),2);
    return retVal; 

def assessAUC(truth,prediction):
    fpr, tpr, thresh = roc_curve(truth,prediction); calcAUC = auc(fpr, tpr); 
    return [fpr,tpr,calcAUC];

def makeROC(inDict,title):
    fig = go.Figure();
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line={'color':'navy','width':2}));
    fig.add_trace(go.Scatter(x=inDict[0], y=inDict[1], line={'color':'darkorange','width':2,'dash':'dash'}));
    fig.update_layout(title=('ROC curve (area = '+str(round(inDict[2],MRL))+') for '+title)
                   ,xaxis_title='False Positive Rate'
                   ,yaxis_title='True Positive Rate')
    return fig;

def checkConfReq(propName,req):
    return (df_Prop[df_Prop.Column==propName][req].iloc[0]==1);

def setDiff(listA,listB):
    return list(set(listA) - set(listB));

def setInt(listA,listB):
    return list(set(listA) & set(listB));
# END LOGIC FUNCTIONS

app.layout = serve_layout();

if __name__ == '__main__':
    app.run_server(host='localhost', port = 8889, debug=True)