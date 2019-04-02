#*************************
#   Created on 06.08.2018
#   author: Bita Khalili
#*************************

import pandas as pd
import numpy as np
import sys
import os
import glob
import math
from copy import deepcopy
import shutil
from optparse import OptionParser
from collections import OrderedDict
from scipy import stats

def Redo(source_dir):
	if (len(glob.glob('original'))>0):
		sig_dirs=glob.glob('*.sig')
		for items in sig_dirs:
			shutil.rmtree(source_dir+'/'+items)
		org_dir=source_dir+'/original/'
		print(os.listdir(org_dir))
		for items in os.listdir(org_dir):
			print(org_dir+items)
			print(source_dir+'/'+items)
			os.renames(org_dir+items,source_dir+'/'+items)
	else:
		print('original directory doesnt exist')
	sys.exit()
def make_newps(scoreadj_dict,scores_dict,shifts,ps_dict,ps_method,adj_score_threshold):
	####### 1. if the max scoreadj for the pseudospetra is above adj_score_threshold with finite metabomatching score add to scoreadj dict
	scoreadjDict={key:scoreadj_dict[key][i] for key in scoreadj_dict for i in range(len(scoreadj_dict[key])) if scoreadj_dict[key][i]>=adj_score_threshold and np.isfinite(scores_dict[key][i])}

	for key in list(scoreadjDict):      # replacing the original tag for those containing .neg or .pos and the highest score in
		if 'neg' in key or 'pos' in key:
			if key.rsplit('.',1)[0] in scoreadjDict:
				scoreadjDict[key.rsplit('.',1)[0]]=max(scoreadjDict[key],scoreadjDict[key.rsplit('.',1)[0]])
			else:
				scoreadjDict[key.rsplit('.',1)[0]]=scoreadjDict[key]
			del scoreadjDict[key]
	############## creating new set of pseudospectra ##############
	newps=[]    #stores pseudospectra that contain at least one metabolite that has a score-adj of score-adj-max
	newps.append(shifts)
	newheaders=[]
	newheaders.append('shift')

	for key in ps_dict:
		if key in scoreadjDict:
			#print(key)
			#print(scoreadj_dict[key])
			newps.append(ps_dict[key])
			newheaders.append(ps_method+'/'+key)

	newps=np.array(newps)
	return (newps,newheaders)

def make_dataframe(metabolites,cas,scoreadj_dict,score_dict,ps_dict,allmethod,source_dir):
	################### creating a dataframe with significant metabolites; at least one hit with scoreadj above 2 and only from pseudospectra with at least one zscore above z-score-threshold #########
	common_keys=[]

	for key in ps_dict:
		if key in scoreadj_dict:
			common_keys.append(key)
			common_keys.append(key+'.pos')
			common_keys.append(key+'.neg')

	#################creating new arrays for scoreadj and score so that they only include scores for the pseudospectra that have at least one zscore above z-score-threshold
	scoreadj_newps=[]
	score_newps=[]
	headers_newps=[]

	for key in common_keys:
		if key in scoreadj_dict:
			scoreadj_newps.append(scoreadj_dict[key])
			score_newps.append(score_dict[key])
			headers_newps.append(key)

	scoreadj_newps=np.array(scoreadj_newps).transpose()
	score_newps=np.array(score_newps).transpose()
	###############Looking for column indices with max scoreadj for all rows 
	argmax_indices=[np.argwhere(scoreadj_newps[i,:]==np.max(scoreadj_newps[i,:])).flatten().tolist() for i in range(scoreadj_newps.shape[0])]   

	###############Taking care of rows with more than one max scoreajd or with nan MM score
	for i in range(len(argmax_indices)):
		if not np.isfinite(score_newps[i,:]).all():
			argmax_indices[i]=[]
		else:
			if len(argmax_indices[i])>1:
				argmax_indices[i]=[item for item in argmax_indices[i] if score_newps[i,item]==max(score_newps[i,argmax_indices[i]])]

	rowmax={}
	rowmax_scoradj={}
	for i in range(len(argmax_indices)):
		if (len(argmax_indices[i])!=0):
			if (len(argmax_indices[i])<2):
				rowmax[metabolites[i]]=[headers_newps[argmax_indices[i][0]]]
				rowmax_scoradj[metabolites[i]]=scoreadj_newps[i][argmax_indices[i][0]] 
			else:
				rowmax[metabolites[i]]=[headers_newps[item] for item in argmax_indices[i]]
				rowmax_scoradj[metabolites[i]]=np.max([scoreadj_newps[i][item] for item in argmax_indices[i]])

	###############Looking for row indices with max scoreadj for all rows 
	for j in range(scoreadj_newps.shape[1]):
		for i in range(scoreadj_newps.shape[0]):
			if not np.isfinite(score_newps[i,j]):
				scoreadj_newps[i,j]=-1
	met_indices=[np.argwhere(scoreadj_newps[:,j]==np.max(scoreadj_newps[:,j])).flatten().tolist() for j in range(scoreadj_newps.shape[1]) ] 
	#print(met_indices)  
	###############Taking care of rows with more than one max scoreajd or with nan MM score
	for j in range(len(met_indices)):
		if len(met_indices[j])>1:
			met_indices[j]=[item for item in met_indices[j] if score_newps[item,j]==max(score_newps[met_indices[j],j])]

	colmax={headers_newps[j]:metabolites[met_indices[j][0]] for j in range(len(met_indices))}
	#print(colmax)		
	df_lists = [] #list of metabolite, ps_tag, scoreadj

	for rowkey in rowmax:
		for colkey in colmax:	
			if rowkey==colmax[colkey] and colkey not in rowmax[rowkey]:
				rowmax[rowkey]=rowmax[rowkey]+[colkey]
				
		df_lists.append([rowkey,rowmax[rowkey],rowmax_scoradj[rowkey]])
	df=pd.DataFrame(df_lists,columns=['metabolites','tags_'+allmethod,'scoreadj_'+allmethod])
	#df.to_csv(source_dir+'/df_'+allmethod+'.tsv',sep='\t')	
	return df

def main(dirs,z_score_threshold,adj_score_threshold,redo_flag):   
	print("\nThe threshold z-score for filtering is:",z_score_threshold,"\nThe threshold adjusted score for filtering is:",adj_score_threshold,"\n")
	source_dir=os.getcwd() 
	if redo_flag:
		Redo(source_dir)
	sigtag = '.sig'
	dirs=[x for x in dirs if x[-len(sigtag):]!=sigtag] ## changed
	print(dirs)
	if len(dirs)>0:
		bestMatchesDF=pd.DataFrame()
		for idir in range(len(dirs)):
			new_dir=os.path.join(source_dir,dirs[idir])
			os.chdir(new_dir)
			print("----------Current directory------------")
			print(os.getcwd())
			allmethod=dirs[idir].split('.',1)[1]
			ps_file=score_file=scoreadj_file=param_file=casname_file=desc_file=tags_file=None
			for directories, folders, files in os.walk('./'):
				for file in files:
					if '.pseudospectrum.tsv' in file:
						ps_file=pd.read_csv(file,header=None,sep='\t')
					if '.score.tsv' in file:
						score_file=pd.read_csv(file,header=None,sep='\t')
					if '.scoreadj.tsv' in file:
						scoreadj_file=pd.read_csv(file,header=None,sep='\t')
					if 'parameters.in.tsv' in file:
						param_file=pd.read_csv(file,header=None,sep='\t')
					if '.casname' in file:
						casname_file=pd.read_csv(file,header=None, sep='\t')
					if 'description.tsv' in file:
						desc_file=pd.read_csv(file,header=None,sep='\t')

			if (not(ps_file is None)) & (not(score_file is None)) & (not(scoreadj_file is None)) & (not(param_file is None)) & (not(casname_file is None)) :
				
				##### if pseudospectra contains at least one peak above z-score-threshold add them to ps_dict#####
				pseudospectrums=ps_headers=ps_method=[]
				pseudospectrums=ps_file.loc[1:,1:].astype(float).values
				ps_headers=ps_file.loc[0,1:].values
				ps_method=ps_headers[0].split('/')[0]
				if 'samplesize' in param_file[0].values:
					samplesize=int(param_file[1].loc[param_file[0]=='samplesize'])
				if 'crscale' in param_file[0].values:
					crscale=float(param_file[1].loc[param_file[0]=='crscale'].values[0])
				else:
					crscale=0

				ps_z=[]
				if ps_method == 'cr':	
					if crscale!=0:
						ps_z=(np.arctanh(pseudospectrums)*math.sqrt(samplesize-3))/crscale
					else:
						ps_z=stats.zscore(np.arctanh(pseudospectrums)*math.sqrt(pseudospectrums.shape[0]-3)) 
				elif ps_method == 'isa' or ps_method == 'PC':
					ps_z=stats.zscore(pseudospectrums)

				ps_dict={ps_headers[k].split('/')[1]:pseudospectrums[:,k] for k in range(pseudospectrums.shape[1]) if (abs(ps_z[:,k])>z_score_threshold).any()}
				# for key in ps_dict:
				# 	print(key)

				scoreadj=scoreadj_file.loc[1:,2:].astype(float).values
				scoreadj_headers=scoreadj_file.loc[0,2:].values   #this one has most number of headers due to .pos .neg headers
				scoreadj_headers=[x.replace('m','M') for x in scoreadj_headers]
				score=score_file.loc[1:,2:].astype(float).values
				score_headers=score_file.loc[0,2:].values
				score_headers=[x.replace('m','M') for x in score_headers]
				shifts=ps_file.loc[1:,0].values
				scoreadj_dict={scoreadj_headers[j].split('/')[1]:scoreadj[:,j] for j in range(scoreadj.shape[1])}
				score_dict={score_headers[j].split('/')[1]:score[:,j] for j in range(score.shape[1])}

				################### MAKE NEWPS ########################
				(newps,newheaders)=make_newps(deepcopy(scoreadj_dict),score_dict,shifts,ps_dict,ps_method,adj_score_threshold)
				if len(newps)>1:    # saving the new pseudospectrum file containing all the pseudospectra that have significant metablite matches 
					newps_df=pd.DataFrame(np.array(newps).transpose())
					outdir=source_dir+'/'+dirs[idir].rsplit('/',1)[0]+sigtag+'/' ## changed
					if not os.path.exists(outdir):
						os.makedirs(outdir)
					newps_df.to_csv(outdir+'/'+dirs[idir].rsplit('/',1)[0]+'.pseudospectrum.tsv',index=False,header=newheaders,sep='\t') ## changed
					shutil.copy(new_dir+'/parameters.in.tsv',outdir+'/parameters.in.tsv')
					if desc_file is not None:
						shutil.copy(new_dir+'/description.tsv',outdir+'/description.tsv')
				else:
					print('didnt find any pseudospectrum in this ps folder')
				if not os.path.exists(source_dir+'/original/'):
					os.makedirs(source_dir+'/original/')
				os.renames(source_dir+'/'+dirs[idir],source_dir+'/original/'+dirs[idir])
				################### END of MAKE NEWPS ########################

				################### Create a dataframe for the table #######################
				cas=score_file.loc[1:,0].values
				casnames=casname_file.values
				metabolites=[]
				for i in range(len(cas)):
					casname_flag=np.logical_not(casnames[:,0]!=cas[i]) 
					metabolites.append(casnames[casname_flag,1][0])
				df=make_dataframe(metabolites,cas,scoreadj_dict,score_dict,ps_dict,allmethod,source_dir)
				bestMatchesDF = pd.concat([bestMatchesDF, df.set_index('metabolites')], axis=1, sort=False)

		bestMatchesDF['sum']=bestMatchesDF.fillna(0).sum(axis=1,numeric_only=True)
		bestMatchesDF=bestMatchesDF.sort_values(by='sum',ascending=False)
		bestMatchesDF.drop(columns=['sum'],inplace=True)
		cols=bestMatchesDF._get_numeric_data().columns
		bestMatchesDF = bestMatchesDF[(bestMatchesDF[cols]>=adj_score_threshold).any(axis=1)]
		bestMatchesDF.to_csv(source_dir+'/MatchesTable.tsv',sep='\t')
	else:
		print('can not find folders starting with ps.')


if __name__ == '__main__':
	dirs=[]
	dirs=glob.glob('ps.*')
	usage = "usage: %prog [optional] arg"
	parser = OptionParser(usage)
	parser.add_option('-r','--redo',dest='redo_flag',default=False)
	parser.add_option('-z','--zscore',dest='z_score_threshold',type='float',default=4)
	parser.add_option('-s','--adjscore',dest='adj_score_threshold',type='float',default=2)
	(options, args) = parser.parse_args()
	main(dirs,options.z_score_threshold,options.adj_score_threshold,options.redo_flag)






