####################################################################################
# Jim Caine
# March, 2015
# CSC529 (Inclass)
# Final Project - Synthetic Sampling Boost / Plankton
# caine.jim@gmail.com
####################################################################################

import os
import random
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import AdaBoostClassifier as AB
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
palette = sns.color_palette()

DATA_PATH = './data/'
CHARTS_PATH = './charts/'


###########################################################################################
# DATA PREPROCESSING
###########################################################################################
def load_data(filename):
	df = pd.DataFrame.from_csv(DATA_PATH + filename)
	return df


def normalize(df):
	# create a list of columns with object datatype
	cols_to_drop = []
	for c in df.columns:
		if df[c].dtype == 'object':
			cols_to_drop.append(c)
	df_objects = df[cols_to_drop]
	df = df.drop(cols_to_drop,axis=1)

	# [0, 1] normalization
	df = (df - df.min()) / (df.max() - df.min())

	# reattach nominal columns and return dataframe
	df = pd.concat([df,df_objects], axis=1)
	return df


def partition_data(df):
	test_perc = 0.3
	test_indices = random.sample(df.index, int(test_perc*df.shape[0]))
	test = df.loc[test_indices]
	train = df.drop(test_indices)
	return train, test

###########################################################################################
# SAMPLING PROCEDURES
###########################################################################################
def create_smote_vectors(df, duplicate_ratio=1.0, proj_distance=1.0, filename='smote'):
	print 'Generating Synethic Vectors (SMOTE)...'

	# remember df column names to apply to smote df at the end
	df_colnames = df.columns

	# find the number of samples that need to be created for each class
	class_numsamples_dict = {}
	class_samples_counts = df.cl.value_counts()
	max_samples = class_samples_counts.iloc[0]
	class_list = df.cl.unique()
	for c in class_list:
		num_samples = class_samples_counts[c]
		class_numsamples_dict[c] = (max_samples-num_samples) / num_samples

	# fit the knn algorithm
	xtrain = df.drop(['cl','filepath'], axis=1)
	nn = NN(n_neighbors=2)
	nn.fit(xtrain)

	# find vector shape
	vshape = xtrain.loc[0].shape

	smote_vectors = []
	# for each instance in the dataframe
	for index, value in df.iterrows():
		# print status update
		if index % 250 == 0:
			print '%s row out of %s' % (str(index), str(df.shape[0]))

		# grab the class
		cl = value.cl

		# eliminate class and filepath to find nn
		x = value.drop(['cl','filepath'])

		# find the nearest neighbor
		distance, indices = nn.kneighbors(x)
		nn_x = df.loc[indices[0][1]].drop(['cl', 'filepath'])

		# generate class_numsamples_dict[class]*duplicate_ratio samples and write to list
		diff = x - nn_x
		for i in range(int(class_numsamples_dict[cl]*duplicate_ratio)):
			r = np.random.random(size=vshape)*proj_distance
			random_difference = diff*r
			v_smote = list(random_difference + x)
			v_smote.append(cl)
			v_smote.append('synthetic')
			v_smote.append(index)
			smote_vectors.append(v_smote)

	# turn smote list into data frame
	df_smote = pd.DataFrame(smote_vectors)
	df_smote.index = df_smote.ix[:,24]
	df_smote = df_smote.drop([24], axis=1)
	df_smote.columns = df_colnames

	# write dataframe to csv
	df_smote.to_csv(DATA_PATH + '%s.csv' % str(filename))


def create_cluster_vectors(df, duplicate_ratio=1.0, proj_distance=1.0, filename='cluster'):
	print 'Generating Synethic Vectors (Cluster)...'

	# remember df column names to apply to smote df at the end
	df_colnames = df.columns

	# find the number of samples that need to be created for each class
	class_numsamples_dict = {}
	class_samples_counts = df.cl.value_counts()
	max_samples = class_samples_counts.iloc[0]
	class_list = df.cl.unique()
	for c in class_list:
		num_samples = class_samples_counts[c]
		class_numsamples_dict[c] = (max_samples-num_samples)/num_samples

	# find the center for each class
	class_clustercenter_dict = {}
	for c in class_list:
		df_c = df[df.cl == c]
		clustercenter = df_c.mean()
		class_clustercenter_dict[c] = clustercenter

	# find vector shape
	vshape = df.drop(['cl','filepath'], axis=1).loc[0].shape

	cluster_vectors = []
	# for each instance in the dataframe
	for index, value in df.iterrows():
		# print status update
		if index % 250 == 0:
			print '%s row out of %s' % (str(index), str(df.shape[0]))

		# grab the class
		cl = value.cl

		# eliminate class and filepath to find nn
		x = value.drop(['cl','filepath'])

		# find the cluster center
		clustercenter = class_clustercenter_dict[cl]

		# generate class_numsamples_dict[class]*duplicate_ratio samples and write to list
		diff = x - clustercenter
		for i in range(int(class_numsamples_dict[cl]*duplicate_ratio)):
			r = np.random.random(size=vshape)*proj_distance
			random_difference = diff*r
			v_cluster = list(random_difference + x)
			v_cluster.append(cl)
			v_cluster.append('synthetic')
			v_cluster.append(index)
			cluster_vectors.append(v_cluster)

	# turn cluster list into data frame
	df_cluster = pd.DataFrame(cluster_vectors)
	df_cluster.index = df_cluster.ix[:,24]
	df_cluster = df_cluster.drop([24], axis=1)
	df_cluster.columns = df_colnames

	# write dataframe to csv
	df_cluster.to_csv(DATA_PATH + '%s.csv' % str(filename))


def create_acluster_vectors(df, duplicate_ratio=1.0, proj_ratio=0.25, filename='acluster'):
	# remember df column names to apply to smote df at the end
	df_colnames = df.columns

	# find the number of samples needed for each class to match majority class
	class_numsamples_dict = {}
	class_samples_counts = df.cl.value_counts()
	max_samples = class_samples_counts.iloc[0]
	class_list = df.cl.unique()
	for c in class_list:
		num_samples = class_samples_counts[c]
		class_numsamples_dict[c] = max_samples-num_samples

	# find vector shape
	vshape = df.drop(['cl','filepath'], axis=1).loc[0].shape

	# find projection lengths
	proj_length = df.std()

	list_syntheticvectors = []
	for c in class_list:
		print c
		# refine dataframe to just the c class
		df_c = df[df.cl == c]

		# find the centroid for class
		centroid = df_c.mean()
		centroid_size = centroid.shape[0]

		# create N synthetic vectors and write to list
		for i in range(int(duplicate_ratio*class_numsamples_dict[c])):
			random_vector = np.random.choice([1,-1], size=centroid_size)*np.random.rand(centroid_size)
			rand_projection = (proj_length*proj_ratio)*random_vector
			v = list((centroid + rand_projection).values)
			v.append(c)
			v.append('synthetic')
			list_syntheticvectors.append(v)

	# turn into a data frame and write to csv
	df_syntheticvectors = pd.DataFrame(list_syntheticvectors)
	df_syntheticvectors.columns = df_colnames
	df_syntheticvectors.to_csv(DATA_PATH + '%s.csv' % filename)


def create_oversample_dataset(df, df_synthetic=None, filename='smote1.csv'):
	df_merge = pd.concat([df,df_synthetic])
	df_merge.to_csv(DATA_PATH + filename)




def generate_stratified_sample(df):
	print 'Generating Stratified Sample'

	# find the median sample size
	class_samples_counts = df.cl.value_counts()
	numclasses = int(np.median(class_samples_counts.values))

	df_ss_indices = []
	for c in df.cl.unique():
		df_c = df[df.cl == c]
		if df_c.shape[0] <= numclasses:
			df_c_indices = df_c.index.values
		else:
			df_c_indices = random.sample(df_c.index, numclasses)
		for i in df_c_indices:
			df_ss_indices.append(i)
	df_ss = df.loc[df_ss_indices]
	df_ss.to_csv(DATA_PATH + 'stratifiedsample.csv')


def convert_cr(cr_string):
	cr = cr_string.split('\n')
	del cr[1]
	cr_temp = []
	for e in cr:
		cr_temp.append(e.split())
	cr = pd.DataFrame(cr_temp)
	cr = cr.drop([5,6], axis=1)
	cr = cr.drop(0)
	cr = cr.drop(cr.tail(3).index)
	cr.columns = ['cl', 'precision', 'recall', 'f1-score', 'support']
	cr['precision'] = cr.precision.astype('float')
	cr['recall'] = cr.recall.astype('float')
	cr['f1-score'] = cr['f1-score'].astype('float')
	cr['support'] = cr.support.astype('int')	

def convert_cr2(cr_string):
	cr = cr_string.split('\n')
	del cr[1]
	cr_temp = []
	for e in cr:
		cr_temp.append(e.split())
	cr = pd.DataFrame(cr_temp)
	cr = cr.drop([5,6], axis=1)
	cr = cr.drop(0)
	cr = cr.drop(cr.tail(3).index)
	cr.columns = ['cl', 'precision', 'recall', 'f1-score', 'support']
	cr['precision'] = cr.precision.astype('float')
	cr['recall'] = cr.recall.astype('float')
	cr['f1-score'] = cr['f1-score'].astype('float')
	cr['support'] = cr.support.astype('int')	







###########################################################################################
# DATA EXPLORATION
###########################################################################################
def plot_histogram_ninstancesperclass(df):
	# create dataframe consisting of the value counts for each target variable
	samples_per_class = df.cl.value_counts()
	print '######################### DATA EXPLORATION #############################'
	print 'Number of classes per species:'
	print samples_per_class
	print '########################################################################'
	print

	# plot in a histogram
	samples_per_class = samples_per_class.values
	plt.hist(samples_per_class, bins=30)
	plt.title('Histogram of Number of Instances per Class')
	plt.xlabel('Number of Instances')
	plt.ylabel('Number of Classes')
	plt.savefig(CHARTS_PATH + 'hist_numinstancesperclass.png')
	plt.clf()


###########################################################################################
# EVALUATION
###########################################################################################
def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    Parameters
    ----------
    y_true : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    species = []
    for f in os.listdir('./train/'):
    	species.append(f)
    if '.DS_Store' in species:
    	species.remove('.DS_Store')
    species_dict = {}
    for i in range(len(species)):
    	species_dict[species[i]] = i
    y_true = y_true.replace(species_dict)
	
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    n_samples = actual.shape[0]
    actual[np.arange(n_samples), y_true.astype(int)] = 1
    vectsum = np.sum(actual * np.log(predictions))
    loss = -1.0 / n_samples * vectsum
    return loss



###########################################################################################
# ML ALGORITHMS
###########################################################################################
class ABM2(object):
	def __init__(self,
				 base_learner=DT(max_depth=2),
				 n_estimators=3,
				 synthetic_data=None,
				 synthetic_ratio=1.0):
		print 'test'
		self.m = base_learner
		self.T = n_estimators
		self.df_synthetic = synthetic_data
		self.synthetic_ratio = synthetic_ratio
		print self.df_synthetic

	def fit(self, x, y):
		# calculate the number of samples per class that need to be generated
		if type(self.df_synthetic) != type(None):
			class_numsamples_dict = {}
			class_samples_counts = y.value_counts()
			max_samples = class_samples_counts.iloc[0]
			class_list = y.unique()
			for c in class_list:
				num_samples = class_samples_counts[c]
				class_numsamples_dict[c] = max_samples-num_samples

		# calculate the number of classes in y
		class_list = y.unique()
		nclasses = len(class_list)
		
		# initialize list to hold models and model weights (alphas)
		self.models = []
		self.alphas = []

		# create copies of x and y to retrieve after training
		xconstant = x.copy()
		yconstant = y.copy()

		# initialize weights
		w0 = 1.0 / x.shape[0]
		w = pd.Series([w0]*x.shape[0], index=x.index)
		w.name = 'w'
		w_index = w.index.values

		# iterate T times
		for i in range(self.T):
			if i == 0:
				xsynthetic = x
				ysynthetic = y
			else:
				print 'identify hardcases'
				# identify hard examples from the original data set for different classes
				new_indices = np.random.choice(w.index,
											   size=xconstant.shape[0],
											   replace=True,
											   p=w.values)
				xsynthetic = xconstant.loc[new_indices]
				ysynthetic = yconstant.loc[new_indices]

				print 'add synthetic cases'
				if type(self.df_synthetic) != type(None):
					# generate synthetic data to balance the training knowledge of different classes
					synthetic_indices = []
					for c in class_numsamples_dict:
						df_c = self.df_synthetic[self.df_synthetic.cl == c]
						synthetic_ind = random.sample(df_c.index, \
							np.minimum(int(class_numsamples_dict[c]*self.synthetic_ratio), df_c.shape[0]))
						for i in synthetic_ind:
							synthetic_indices.append(i)
					synthetic_samples = self.df_synthetic.loc[synthetic_indices]
					x_synthetic_samples = synthetic_samples.drop(['cl','filepath'], axis=1)
					y_synthetic_samples = synthetic_samples.cl
					xsynthetic = pd.concat([x_synthetic_samples, xsynthetic])
					ysynthetic = pd.concat([y_synthetic_samples, ysynthetic])

				# ensure that every class is represented in new dataframe
				if len(ysynthetic.unique()) != 121:
					print 'ensure every class is represented'
					non_represented_classes = np.setdiff1d(class_list, ysynthetic.unique())
					for c in non_represented_classes:
						index = random.sample(yconstant[yconstant == c].index, 1)
						xunderrep = xconstant.loc[index]
						yunderrep = yconstant.loc[index]
						xsynthetic = pd.concat([xsynthetic, xunderrep], axis=0)
						ysynthetic = pd.concat([ysynthetic, yunderrep])

			print 'train'
			# train a weak learner
			clf = self.m.fit(xsynthetic, ysynthetic)
			self.models.append(clf)

			print 'pred'
			# get back the hypothesis
			pred = clf.predict(xconstant)

			# calculate the error
			df_err = pd.DataFrame({'pred':pred, 'actual':y})
			h_t = (df_err['pred'] != df_err['actual'])*1
			e = 0.5*h_t.values.dot(w.values).sum()

			# update alpha
			alpha = np.log((1-e)/e)
			self.alphas.append(alpha)

			# update weights
			h = h_t.replace(0, -1).values
			w = w.values*(np.exp(-alpha*h))
			w = w / w.sum()
			w = pd.Series(w, index=w_index)

	def predict(self, x):
		# iterate through each model and stuff predictions in predictions_df
		predictions = []
		for m in self.models:
			# create a prediction
			pred = m.predict(x)
			predictions.append(pred)
		predictions = pd.DataFrame(predictions).transpose()

		# predict the class with the largest sum of alphas
		weighted_predictions = []
		for index, value in predictions.iterrows():
			predictions_prob_dict = {}
			for i in range(len(self.alphas)):
				v = value.iloc[i]
				if v not in predictions_prob_dict.keys():
					predictions_prob_dict[v] = self.alphas[i]
				else:
					predictions_prob_dict[v] += self.alphas[i]
			weighted_predictions.append(max(predictions_prob_dict, key=predictions_prob_dict.get))
		return weighted_predictions

	def predict_proba(self, x):
		predictions = []
		for m in self.models:
			pred_proba = m.predict_proba(x)
			predictions.append(pred_proba)
		weighted_predictions = []
		for i in range(len(predictions)):
			weighted_predictions.append(self.alphas[i]*predictions[i])
		predict_proba = sum(weighted_predictions)
		return predict_proba


class ABM3(object):
	def __init__(self, k=5.0):
		self.k = k

	def fit(self, x, y):
		# calculate sample ratio
		class_numsamples_dict = {}
		class_samples_counts = y.value_counts()
		max_samples = class_samples_counts.iloc[0]
		class_list = y.unique()
		for c in class_list:
			num_samples = class_samples_counts[c]
			class_numsamples_dict[c] = float(max_samples-num_samples)/max_samples
		sample_ratio = []
		for index, value in y.iteritems():
			c = value
			sample_ratio.append(class_numsamples_dict[c])

		# calculate knn ratio
		x = x.reset_index()
		y = y.reset_index()
		nn = NN(n_neighbors=int(self.k))
		nn.fit(x)
		knn_ratio = []
		for index, value in x.iterrows():
			if index % 100 == 0:
				print index
			num_sameclass = 0
			distance, indices = nn.kneighbors(value)
			relevant_indices = indices[0][1:]
			for ri in relevant_indices:
				if y.loc[ri].values[1] == y.loc[index].values[1]:
					num_sameclass += 1
			knn_ratio.append(num_sameclass / float(self.k))
		plt.hist(knn_ratio)
		plt.show()

		#


		# initialize weights

		# iterate T times
			# identify hard examples

			# ensure every class is represented in the sample

			# train a weak learner

			# get the hypothesis

			# calculate error

			# update alpha

			# update weights


class rusBoost(object):
	def __init__(self, base_learner=DT(max_depth=2),
				 n_estimators=3,
				 rus_ratio=1.0,
				 class_numsamples_dict=False):
		self.m = base_learner
		self.T = n_estimators
		self.rus_ratio = rus_ratio
		self.class_numsamples_dict = class_numsamples_dict

	def fit(self, xtrain, ytrain):
		# initialize list to hold models and model weights (alphas)
		self.models = []
		self.alphas = []

		xtrain_constant = xtrain.copy()
		ytrain_constant = ytrain.copy()

		# initialize weights
		w0 = 1.0 / xtrain.shape[0]
		w = pd.Series([w0]*xtrain.shape[0], index=xtrain.index)
		w.name = 'w'

		# iterate T times
		for i in range(self.T):
			# keep track of the index on the weights vector
			w_index = w.index.values

			# modify the distribution by performing RUS
			ind_to_keep = []
			for c in self.class_numsamples_dict:
				df_c = ytrain[ytrain == c]
				rus_ind = random.sample(df_c.index, np.minimum(self.rus_ratio, df_c.shape[0]))
				for i in rus_ind:
					ind_to_keep.append(i)
			x_rus = xtrain.loc[ind_to_keep]
			y_rus = ytrain.loc[ind_to_keep]
			# train a weak learner
			clf = self.m.fit(x_rus, y_rus)
			self.models.append(clf)

			# make predictions and compute loss
			pred = clf.predict(xtrain_constant)
			df_err = pd.DataFrame({'pred':pred, 'actual':ytrain})
			h_t = (df_err['pred'] != df_err['actual'])*1
			e = 0.5*h_t.values.dot(w.values).sum()

			# update alpha
			alpha = np.log((1-e)/e)
			self.alphas.append(alpha)

			# update weights
			h = h_t.replace(0, -1).values
			w = w.values*(np.exp(-alpha*h))
			w = w / w.sum()
			w = pd.Series(w, index=w_index)			

			# update the data frame
			new_indices = np.random.choice(w.index,
										   size=xtrain_constant.shape[0],
										   replace=True,
										   p=w.values)
			xtrain = xtrain_constant.loc[new_indices]
			ytrain = ytrain_constant.loc[new_indices]


	def predict(self, x):
		# iterate through each model and stuff predictions in predictions_df
		predictions = []
		for m in self.models:
			# create a prediction
			pred = m.predict(x)
			predictions.append(pred)
		predictions = pd.DataFrame(predictions).transpose()

		# predict the class with the largest sum of alphas
		weighted_predictions = []
		for index, value in predictions.iterrows():
			predictions_prob_dict = {}
			for i in range(len(self.alphas)):
				v = value.iloc[i]
				if v not in predictions_prob_dict.keys():
					predictions_prob_dict[v] = self.alphas[i]
				else:
					predictions_prob_dict[v] += self.alphas[i]
			weighted_predictions.append(max(predictions_prob_dict, key=predictions_prob_dict.get))

		return weighted_predictions

	def predict_proba(self, x):
		predictions = []
		for m in self.models:
			pred_proba = m.predict_proba(x)
			predictions.append(pred_proba)
		weighted_predictions = []
		for i in range(len(predictions)):
			weighted_predictions.append(self.alphas[i]*predictions[i])
		predict_proba = sum(weighted_predictions)
		return predict_proba




###########################################################################################
# ANALYSIS
###########################################################################################
class adaBoostAnalysis():
	def __init__(self, train, test):
		self.train = train
		self.test = test
		self.xtrain = train.drop(['cl', 'filepath'], axis=1)
		self.ytrain = train.cl
		self.xtest = test.drop(['cl', 'filepath'], axis=1)
		self.ytest = test.cl

	def iterate(self, n_estimators_conf=[10], learning_rate_conf=[0.25]):
		print '-'*80
		print 'Running AdaBoost Iterations...'
		# performance by number of estimators and max depth
		results = []
		for ne in n_estimators_conf:
			for lr in learning_rate_conf:
				print 'Iteration: n_estimators=%s, learning_rate=%s' % (str(ne), str(lr))
				m = AB(n_estimators=ne, learning_rate=lr)
				m.fit(self.xtrain, self.ytrain)
				predtrain = m.predict(self.xtrain)
				predtest = m.predict(self.xtest)
				predprobatrain = m.predict_proba(self.xtrain)
				predprobatest = m.predict_proba(self.xtest)
				accuracytrain = metrics.accuracy_score(predtrain, self.ytrain)
				accuracytest = metrics.accuracy_score(predtest, self.ytest)
				kstrain = multiclass_log_loss(self.ytrain, predprobatrain)
				kstest = multiclass_log_loss(self.ytest, predprobatest)
				cr = self.convert_cr(metrics.classification_report(self.ytest, predtest))
				results.append([ne, lr, accuracytrain, accuracytest, kstrain, kstest, cr])
		self.results = pd.DataFrame(results)
		self.results.columns = ['ne', 'lr', 'accuracy_train', 'accuracy_test',
						   'ks_train', 'ks_test', 'cr']

	def convert_cr(self, cr_string):
		cr = cr_string.split('\n')
		del cr[1]
		cr_temp = []
		for e in cr:
			cr_temp.append(e.split())
		cr = pd.DataFrame(cr_temp)
		cr = cr.drop([5,6], axis=1)
		cr = cr.drop(0)
		cr = cr.drop(cr.tail(3).index)
		cr.columns = ['cl', 'precision', 'recall', 'f1-score', 'support']
		cr['precision'] = cr.precision.astype('float')
		cr['recall'] = cr.recall.astype('float')
		cr['f1-score'] = cr['f1-score'].astype('float')
		cr['support'] = cr.support.astype('int')
		return cr	


class randomForestAnalysis():
	def __init__(self, train, test):
		self.train = train
		self.test = test
		self.xtrain = train.drop(['cl', 'filepath'], axis=1)
		self.ytrain = train.cl
		self.xtest = test.drop(['cl', 'filepath'], axis=1)
		self.ytest = test.cl

	def iterate(self, n_estimators_conf=[10], max_depth_conf=[8]):
		print '-'*80
		print 'Running RandomForest Iterations...'
		# performance by number of estimators and max depth
		results = []
		for ne in n_estimators_conf:
			for md in max_depth_conf:
				print 'Iteration: n_estimators=%s, max_depth=%s' % (str(ne), str(md))
				m = RF(n_estimators=ne, max_depth=md)
				m.fit(self.xtrain, self.ytrain)
				predtrain = m.predict(self.xtrain)
				predtest = m.predict(self.xtest)
				predprobatrain = m.predict_proba(self.xtrain)
				predprobatest = m.predict_proba(self.xtest)
				accuracytrain = metrics.accuracy_score(predtrain, self.ytrain)
				accuracytest = metrics.accuracy_score(predtest, self.ytest)
				kstrain = multiclass_log_loss(self.ytrain, predprobatrain)
				kstest = multiclass_log_loss(self.ytest, predprobatest)
				cr = self.convert_cr(metrics.classification_report(self.ytest, predtest))
				results.append([ne, md, accuracytrain, accuracytest, kstrain, kstest, cr])
		self.results = pd.DataFrame(results)
		self.results.columns = ['ne', 'md', 'accuracy_train', 'accuracy_test',
						   'ks_train', 'ks_test', 'cr']

	def convert_cr(self, cr_string):
		cr = cr_string.split('\n')
		del cr[1]
		cr_temp = []
		for e in cr:
			cr_temp.append(e.split())
		cr = pd.DataFrame(cr_temp)
		cr = cr.drop([5,6], axis=1)
		cr = cr.drop(0)
		cr = cr.drop(cr.tail(3).index)
		cr.columns = ['cl', 'precision', 'recall', 'f1-score', 'support']
		cr['precision'] = cr.precision.astype('float')
		cr['recall'] = cr.recall.astype('float')
		cr['f1-score'] = cr['f1-score'].astype('float')
		cr['support'] = cr.support.astype('int')
		return cr


class rusBoostAnalysis():
	def __init__(self, train, test,
				 n_estimators_conf=[5],
				 rus_ratio_conf=[100]):
		self.train = train
		self.test = test
		self.xtrain = train.drop(['cl', 'filepath'], axis=1)
		self.ytrain = train.cl
		self.xtest = test.drop(['cl', 'filepath'], axis=1)
		self.ytest = test.cl
		self.n_estimators_conf = n_estimators_conf
		self.rus_ratio_conf = rus_ratio_conf

		# calculate the number of samples per class that need to be generated
		class_numsamples_dict = {}
		class_samples_counts = self.ytrain.value_counts()
		max_samples = class_samples_counts.iloc[0]
		class_list = self.ytrain.unique()
		for c in class_list:
			num_samples = class_samples_counts[c]
			class_numsamples_dict[c] = max_samples-num_samples
		self.class_numsamples_dict = class_numsamples_dict

	def iterate(self):
		print '-'*80
		print 'Running RUSBoost Iterations...'
		# performance by number of estimators and max depth
		results = []

		for ne in self.n_estimators_conf:
			for rr in self.rus_ratio_conf:
				print 'Iteration: nestimators=%s, rus_ratio=%s' % (str(ne), str(rr))
				m = rusBoost(base_learner=DT(max_depth=2), n_estimators=ne, rus_ratio=rr,
					   class_numsamples_dict=self.class_numsamples_dict)
				m.fit(self.xtrain, self.ytrain)
				predtrain = m.predict(self.xtrain)
				predtest = m.predict(self.xtest)
				predprobatrain = m.predict_proba(self.xtrain)
				predprobatest = m.predict_proba(self.xtest)
				accuracytrain = metrics.accuracy_score(predtrain, self.ytrain)
				accuracytest = metrics.accuracy_score(predtest, self.ytest)
				kstrain = multiclass_log_loss(self.ytrain, predprobatrain)
				kstest = multiclass_log_loss(self.ytest, predprobatest)
				cr = self.convert_cr(metrics.classification_report(self.ytest, predtest))
				results.append([ne, rr, accuracytrain, accuracytest, kstrain, kstest, cr])

		self.results = pd.DataFrame(results)
		self.results.columns = ['ne', 'rr', 'accuracy_train', 'accuracy_test',
						   'ks_train', 'ks_test', 'cr']

	def convert_cr(self, cr_string):
		cr = cr_string.split('\n')
		del cr[1]
		cr_temp = []
		for e in cr:
			cr_temp.append(e.split())
		cr = pd.DataFrame(cr_temp)
		cr = cr.drop([5,6], axis=1)
		cr = cr.drop(0)
		cr = cr.drop(cr.tail(3).index)
		cr.columns = ['cl', 'precision', 'recall', 'f1-score', 'support']
		cr['precision'] = cr.precision.astype('float')
		cr['recall'] = cr.recall.astype('float')
		cr['f1-score'] = cr['f1-score'].astype('float')
		cr['support'] = cr.support.astype('int')
		return cr


class smoteBoostAnalysis():
	def __init__(self, train, test, class_numsamples_dict, df_smote,
				 n_estimators_conf=[5],
				 smote_ratio_conf=[100]):
		self.train = train
		self.test = test
		self.xtrain = train.drop(['cl', 'filepath'], axis=1)
		self.ytrain = train.cl
		self.xtest = test.drop(['cl', 'filepath'], axis=1)
		self.ytest = test.cl
		self.class_numsamples_dict = class_numsamples_dict
		self.df_smote = df_smote
		self.n_estimators_conf = n_estimators_conf
		self.smote_ratio_conf = smote_ratio_conf

	def iterate(self):
		print '-'*80
		print 'Running SMOTEBoost Iterations...'
		# performance by number of estimators and max depth
		results = []

		for ne in self.n_estimators_conf:
			for sr in self.smote_ratio_conf:
				print 'Iteration: nestimators=%s, smote_ratio=%s' % (str(ne), str(sr))
				m = SB(base_learner=DT(max_depth=2), n_estimators=ne, smote_ratio=sr,
					   class_numsamples_dict=class_numsamples_dict, df_smote=df_smote)
				m.fit(self.xtrain, self.ytrain)
				predtrain = m.predict(self.xtrain)
				predtest = m.predict(self.xtest)
				predprobatrain = m.predict_proba(self.xtrain)
				predprobatest = m.predict_proba(self.xtest)
				accuracytrain = metrics.accuracy_score(predtrain, self.ytrain)
				accuracytest = metrics.accuracy_score(predtest, self.ytest)
				kstrain = multiclass_log_loss(self.ytrain, predprobatrain)
				kstest = multiclass_log_loss(self.ytest, predprobatest)
				results.append([ne, sr, accuracytrain, accuracytest, kstrain, kstest])

		self.results = pd.DataFrame(results)
		self.results.columns = ['ne', 'sr', 'accuracy_train', 'accuracy_test',
						   'ks_train', 'ks_test']




class dataBoostIMAnalysis():
	def __init__(self, train, test):
		self.train = train
		self.test = test
		self.xtrain = train.drop(['cl', 'filepath'], axis=1)
		self.ytrain = train.cl
		self.xtest = test.drop(['cl', 'filepath'], axis=1)
		self.ytest = test.cl

	def iterate(self):
		print '-'*80
		print 'Running DataBoost-IM Iterations...'

		results = []
		m = DBIM(num_iterations=7)
		m.fit(self.xtrain, self.ytrain)
		predtrain = m.predict(self.xtrain)
		predtest = m.predict(self.xtest)
		predprobatrain = m.predict_proba(self.xtrain)
		predprobatest = m.predict_proba(self.xtest)
		accuracytrain = metrics.accuracy_score(predtrain, self.ytrain)
		accuracytest = metrics.accuracy_score(predtest, self.ytest)
		kstrain = multiclass_log_loss(self.ytrain, predprobatrain)
		kstest = multiclass_log_loss(self.ytest, predprobatest)
		results.append([accuracytrain, accuracytest, kstrain, kstest])

		self.results = pd.DataFrame(results)
		self.results.columns = ['accuracy_train', 'accuracy_test',
						   'ks_train', 'ks_test']


class ABM2Analysis():
	def __init__(self,
				 train,
				 test,
				 base_learners_conf = [DT(max_depth=2)],
				 n_estimators_conf=[3],
				 synthetic_data_conf = [None],
				 synthetic_ratio_conf = [1.0]):
		self.base_learners_conf = base_learners_conf
		self.n_estimators_conf = n_estimators_conf
		self.synthetic_data_conf = synthetic_data_conf
		self.synthetic_ratio_conf = synthetic_ratio_conf
		self.train = train
		self.test = test
		self.xtrain = self.train.drop(['cl','filepath'], axis=1)
		self.ytrain = self.train.cl
		self.xtest = self.test.drop(['cl','filepath'], axis=1)
		self.ytest = self.test.cl

	def iterate(self):
		print '-'*80
		print 'Running ABM2 Iterations...'

		results = []
		for bl in self.base_learners_conf:
			for ne in self.n_estimators_conf:
				for sd in self.synthetic_data_conf:
					for sr in self.synthetic_ratio_conf:
						print 'Iteration: bl=%s, ne=%s, sd=%s, sr=%s' % \
							(str(bl), str(ne), str(sd[0]), str(sr))
						m = ABM2(base_learner=bl,
								 n_estimators=ne,
								 synthetic_data=sd[1],
								 synthetic_ratio=sr)
						m.fit(self.xtrain, self.ytrain)
						predtrain = m.predict(self.xtrain)
						predtest = m.predict(self.xtest)
						predprobatrain = m.predict_proba(self.xtrain)
						predprobatest = m.predict_proba(self.xtest)
						accuracytrain = metrics.accuracy_score(predtrain, self.ytrain)
						accuracytest = metrics.accuracy_score(predtest, self.ytest)
						kstrain = multiclass_log_loss(self.ytrain, predprobatrain)
						kstest = multiclass_log_loss(self.ytest, predprobatest)
						cr = self.convert_cr(metrics.classification_report(self.ytest, predtest))
						results.append([bl, ne, sd[0], sr, accuracytrain, accuracytest,
									    kstrain, kstest, cr])
		self.results = pd.DataFrame(results)
		self.results.columns = ['bl', 'ne', 'sd', 'sr', 'acctrain', 'acctest',
						   'kstrain', 'kstest', 'cr']
		print self.results

	def convert_cr(self, cr_string):
		cr = cr_string.split('\n')
		del cr[1]
		cr_temp = []
		for e in cr:
			cr_temp.append(e.split())
		cr = pd.DataFrame(cr_temp)
		cr = cr.drop([5,6], axis=1)
		cr = cr.drop(0)
		cr = cr.drop(cr.tail(3).index)
		cr.columns = ['cl', 'precision', 'recall', 'f1-score', 'support']
		cr['precision'] = cr.precision.astype('float')
		cr['recall'] = cr.recall.astype('float')
		cr['f1-score'] = cr['f1-score'].astype('float')
		cr['support'] = cr.support.astype('int')
		return cr





def plot_random_injection_performance(dt_accuracy, df_results):
	m2ks = df_results[df_results.sd == 'm2'].kstest.values
	smoteks = df_results[df_results.sd == 'smote'].kstest.values
	clusterks = df_results[df_results.sd == 'cluster'].kstest.values
	clusterdiffks = df_results[df_results.sd == 'clusterdiff'].kstest.values
	dtks = [dt_accuracy]*m2ks.shape[0]
	nestimators = df_results[df_results.sd == 'm2']['ne'].values

	plt.plot(nestimators,m2ks,color='r',label='AdaBoost M2')
	plt.plot(nestimators,smoteks,color='b',label='SMOTEBoost')
	plt.plot(nestimators,clusterks,color='g',label='ABM2 Cluster')
	plt.plot(nestimators,clusterdiffks,color='y',label='ABM2 Cluster Diff')
	plt.plot(nestimators,dtks,color='black', label='Benchmark (DT)')
	plt.legend(loc='best')
	plt.title('Log Loss Mean At Varying Number of Estimators')
	plt.xlabel('Number of Estimators')
	plt.ylabel('Log Loss Mean')
	plt.show()
	# df_results = pd.DataFrame({'bl':['dt','dt','dt','dt','dt','dt','dt','dt'],
	# 						   'ne':[2,5,2,5,2,5,2,5],
	# 						   'sd':['m2','m2','smote','smote','cluster','cluster','clusterdiff','clusterdiff'],
	# 						   'kstest':[1,2,3.5,4,5,4,4.5,4.25]})
	# plot_random_injection_performance(3.5, df_results)


def random_injection_analysis():
	# AdaBoost M2
	abm2 = ABM2Analysis(train = train,
						test = test,
						n_estimators_conf = [3,5],
						synthetic_ratio_conf = [0.01],
						synthetic_data_conf=[('m2',None)])
	abm2.iterate()
	abm2results = abm2.results
	# SMOTEBoost
	smote = ABM2Analysis(train = train,
						test = test,
						n_estimators_conf = [3,5],
						synthetic_ratio_conf = [0.001],
						synthetic_data_conf=[('smote',df_smote)])
	smote.iterate()
	smoteresults = smote.results
	# ClusterBoost
	cluster = ABM2Analysis(train = train,
						test = test,
						n_estimators_conf = [3,5],
						synthetic_ratio_conf = [0.001],
						synthetic_data_conf=[('cluster',df_acluster)])
	cluster.iterate()
	clusterresults = cluster.results
	# ClusterBoost
	clusterdiff = ABM2Analysis(train = train,
						test = test,
						n_estimators_conf = [3,5],
						synthetic_ratio_conf = [0.01],
						synthetic_data_conf=[('clusterdiff',df_cluster)])
	clusterdiff.iterate()
	clusterdiffresults = clusterdiff.results

	df_results = pd.concat([abm2results, smoteresults, clusterresults, clusterdiffresults])

	plot_random_injection_performance(3.8, df_results)


def plot_nobservations_vs_recall_chart(cr, title):
	bins = np.linspace(0, cr.support.max(), 20)
	cr_supportbin = cr.groupby(np.digitize(cr.support,bins)).mean()
	plt.plot(range(cr_supportbin.shape[0]), cr_supportbin.recall)
	plt.plot(range(cr_supportbin.shape[0]), cr_supportbin.precision)
	plt.plot(range(cr_supportbin.shape[0]), cr_supportbin['f1-score'])
	plt.title(title)
	plt.xlabel('Support')
	plt.ylabel('Confusion Performance')
	plt.legend(labels=['Recall','Precision', 'F1-Score'], loc='best')
	plt.xticks(range(cr_supportbin.shape[0]), bins.astype('int'))
	plt.savefig(CHARTS_PATH + '%s.png' % title)
	plt.clf()


def analyze_results_rf():
	rfsmote05results = pd.DataFrame.from_csv('rfsmote05results.csv')
	rfcluster05results = pd.DataFrame.from_csv('rfcluster05results.csv')
	rfacluster05results = pd.DataFrame.from_csv('rfacluster05results.csv')
	rfresults = pd.concat([rfsmote05results, rfcluster05results, rfacluster05results])

	smote = rfsmote05results[rfsmote05results.md == 20]
	cluster = rfcluster05results[rfcluster05results.md == 20]
	acluster = rfacluster05results[rfacluster05results.md == 20]
	plt.plot(smote['ne'], smote.ks_test, c=palette[0], label='SMOTEBoost')
	plt.plot(cluster['ne'], cluster.ks_test, c=palette[1], label='PFITCC')
	plt.plot(acluster['ne'], acluster.ks_test, c=palette[2], label='PFCC')
	plt.legend(loc='best')
	plt.xlabel('Number of Trees In Forest')
	plt.ylabel('MC Log Loss')
	plt.title('Random Forest Performance At Varying Number of Trees (depth=20, proj_ratio=0.5)')
	plt.savefig(CHARTS_PATH + 'rfnestimators.png')
	plt.clf()

	smote_md10 = rfsmote05results[rfsmote05results.md == 10]
	smote_md20 = rfsmote05results[rfsmote05results.md == 20]
	smote_md30 = rfsmote05results[rfsmote05results.md == 30]
	plt.plot(smote_md10['ne'], smote_md10['ks_test'], c=palette[0], label='max_depth=10')
	plt.plot(smote_md20['ne'], smote_md20['ks_test'], c=palette[1], label='max_depth=20')
	plt.plot(smote_md30['ne'], smote_md30['ks_test'], c=palette[2], label='max_depth=30')
	plt.legend(loc='best')
	plt.xlabel('Number of Trees In Forest')
	plt.ylabel('MC Log Loss')
	plt.title('Random Forest Performance At Varying Number of Trees And Max Depths (SMOTE proj_ratio=0.5')
	plt.savefig(CHARTS_PATH + 'rfsmote_varyingdepths.png')
	plt.clf()

	rfresults = pd.DataFrame.from_csv('rfresults.csv')
	print rfresults
	rfresults_md2 = rfresults[rfresults.md == 2]
	rfresults_md5 = rfresults[rfresults.md == 5]
	rfresults_md8 = rfresults[rfresults.md == 8]
	rfresults_md12 = rfresults[rfresults.md == 12]
	rfresults_md20 = rfresults[rfresults.md == 20]
	plt.plot(rfresults_md2['ne'], rfresults_md2['ks_test'], c=palette[0], label='max_depth=2')
	plt.plot(rfresults_md2['ne'], rfresults_md5['ks_test'], c=palette[1], label='max_depth=5')
	plt.plot(rfresults_md2['ne'], rfresults_md8['ks_test'], c=palette[2], label='max_depth=8')
	plt.plot(rfresults_md2['ne'], rfresults_md12['ks_test'], c=palette[3], label='max_depth=12')
	plt.plot(rfresults_md2['ne'], rfresults_md20['ks_test'], c=palette[4], label='max_depth=20')
	plt.legend(loc='best')
	plt.xlabel('Number of Trees In Forest')
	plt.ylabel('MC Log Loss')
	plt.title('Random Forest Performance At Varying Number of Trees and Max Depths (No Sample)')
	plt.savefig(CHARTS_PATH +  'rfnosample_varyingdepths.png')
	plt.clf()



###########################################################################################
# DRIVER
###########################################################################################
# load the initial dataset
df = load_data('dataset.csv')
df = normalize(df)
train, test = partition_data(df)

# create synthetic examples
create_smote_vectors(df, duplicate_ratio=1.0, proj_distance=1.0, filename='smote1')
create_smote_vectors(df, duplicate_ratio=1.0, proj_distance=0.1, filename='smote01')
create_smote_vectors(df, duplicate_ratio=1.0, proj_distance=0.5, filename='smote05')
create_smote_vectors(df, duplicate_ratio=1.0, proj_distance=2.0, filename='smote2')
create_smote_vectors(df, duplicate_ratio=1.0, proj_distance=3.0, filename='smote3')
create_cluster_vectors(df, duplicate_ratio=1.0, proj_distance=1.0, filename='cluster1')
create_cluster_vectors(df, duplicate_ratio=1.0, proj_distance=0.1, filename='cluster01')
create_cluster_vectors(df, duplicate_ratio=1.0, proj_distance=0.5, filename='cluster05')
create_cluster_vectors(df, duplicate_ratio=1.0, proj_distance=2.0, filename='cluster2')
create_cluster_vectors(df, duplicate_ratio=1.0, proj_distance=3.0, filename='cluster3')
create_acluster_vectors(df, duplicate_ratio=1.0, proj_ratio=1.0, filename='acluster1')
create_acluster_vectors(df, duplicate_ratio=1.0, proj_ratio=0.1, filename='acluster01')
create_acluster_vectors(df, duplicate_ratio=1.0, proj_ratio=0.5, filename='acluster05')
create_acluster_vectors(df, duplicate_ratio=1.0, proj_ratio=2.0, filename='acluster2')
create_acluster_vectors(df, duplicate_ratio=1.0, proj_ratio=3.0, filename='acluster3')

# load synthetic samples
df_smote1 = load_data('smote1.csv')
df_smote2 = load_data('smote2.csv')
df_smote3 = load_data('smote3.csv')
df_smote01 = load_data('smote01.csv')
df_smote05 = load_data('smote05.csv')
df_cluster1 = load_data('cluster1.csv')
df_cluster2 = load_data('cluster2.csv')
df_cluster3 = load_data('cluster3.csv')
df_cluster01 = load_data('cluster01.csv')
df_cluster05 = load_data('cluster05.csv')
df_acluster1 = load_data('acluster1.csv')
df_acluster2 = load_data('acluster2.csv')
df_acluster3 = load_data('acluster3.csv')
df_acluster01 = load_data('acluster01.csv')
df_acluster05 = load_data('acluster05.csv')

# create synthetic oversampled datasets
create_oversample_dataset(df, df_smote1, filename='synovs_smote1.csv')
create_oversample_dataset(df, df_smote2, filename='synovs_smote2.csv')
create_oversample_dataset(df, df_smote3, filename='synovs_smote3.csv')
create_oversample_dataset(df, df_smote01, filename='synovs_smote01.csv')
create_oversample_dataset(df, df_smote05, filename='synovs_smote05.csv')

create_oversample_dataset(df, df_cluster1, filename='synovs_cluster1.csv')
create_oversample_dataset(df, df_cluster2, filename='synovs_cluster2.csv')
create_oversample_dataset(df, df_cluster3, filename='synovs_cluster3.csv')
create_oversample_dataset(df, df_cluster01, filename='synovs_cluster01.csv')
create_oversample_dataset(df, df_cluster05, filename='synovs_cluster05.csv')

create_oversample_dataset(df, df_acluster1, filename='synovs_acluster1.csv')
create_oversample_dataset(df, df_acluster2, filename='synovs_acluster2.csv')
create_oversample_dataset(df, df_acluster3, filename='synovs_acluster3.csv')
create_oversample_dataset(df, df_acluster01, filename='synovs_acluster01.csv')
create_oversample_dataset(df, df_acluster05, filename='synovs_acluster05.csv')

load oversampled datasets
df_ovs_smote1 = pd.DataFrame.from_csv(DATA_PATH + 'synovs_smote1.csv')
df_ovs_smote2 = pd.DataFrame.from_csv(DATA_PATH + 'synovs_smote2.csv')
df_ovs_smote3 = pd.DataFrame.from_csv(DATA_PATH + 'synovs_smote3.csv')
df_ovs_smote01 = pd.DataFrame.from_csv(DATA_PATH + 'synovs_smote01.csv')
df_ovs_smote05 = pd.DataFrame.from_csv(DATA_PATH + 'synovs_smote05.csv')
df_ovs_cluster1 = pd.DataFrame.from_csv(DATA_PATH + 'synovs_cluster1.csv')
df_ovs_cluster2 = pd.DataFrame.from_csv(DATA_PATH + 'synovs_cluster2.csv')
df_ovs_cluster3 = pd.DataFrame.from_csv(DATA_PATH + 'synovs_cluster3.csv')
df_ovs_cluster01 = pd.DataFrame.from_csv(DATA_PATH + 'synovs_cluster01.csv')
df_ovs_cluster05 = pd.DataFrame.from_csv(DATA_PATH + 'synovs_cluster05.csv')

df_ovs_acluster1 = pd.DataFrame.from_csv(DATA_PATH + 'synovs_acluster1.csv')
df_ovs_acluster2 = pd.DataFrame.from_csv(DATA_PATH + 'synovs_acluster2.csv')
df_ovs_acluster3 = pd.DataFrame.from_csv(DATA_PATH + 'synovs_acluster3.csv')
df_ovs_acluster01 = pd.DataFrame.from_csv(DATA_PATH + 'synovs_acluster01.csv')
df_ovs_acluster05 = pd.DataFrame.from_csv(DATA_PATH + 'synovs_acluster05.csv')



# adaboost
ab = adaBoostAnalysis(train, test)
ab.iterate(n_estimators_conf=[5,10,20,30,40,50],
		   learning_rate_conf=[0.05, .1, .25, .5, 1.0])
ab.results.to_csv('abresults.csv')


# random forest
rf = randomForestAnalysis(train, test)
rf.iterate(n_estimators_conf=[5,10,50,100,500,1000],
		   max_depth_conf=[2,5,8,12,20])
rf.results.to_csv('rfresults.csv')


# rusboost
rb = rusBoostAnalysis(train, test,
					  n_estimators_conf=[5,10,50,100],
					  rus_ratio_conf=[5,10,50,100])
rb.iterate()
rb.results.to_csv('rbresults.csv')

# smoteboost
sb = ABM2Analysis(train = train,
					test = test,
					n_estimators_conf = [2,5,10,20],
					synthetic_ratio_conf = [.01,.1,1],
					synthetic_data_conf=[('m2',None),
										 ('smote1',df_smote1),
										 ('smote2',df_smote2),
										 ('smote3',df_smote3),
										 ('smote01',df_smote01),
										 ('smote05',df_smote05)])
sb.iterate()
sb.results.to_csv('sbresults.csv')

# clusterboost
cb = ABM2Analysis(train = train,
					test = test,
					n_estimators_conf = [2,5,10,20],
					synthetic_ratio_conf = [.01,.1,1],
					synthetic_data_conf=[('cluster1',df_cluster1),
										 ('cluster2',df_cluster2),
										 ('cluster3',df_cluster3),
										 ('cluster01',df_cluster01),
										 ('cluster05',df_cluster05)])
cb.iterate()
cb.results.to_csv('cbresults.csv')

# aclusterboost
acb = ABM2Analysis(train = train,
					test = test,
					n_estimators_conf = [2,5,10,20],
					synthetic_ratio_conf = [.01,.1,1],
					synthetic_data_conf=[('acluster1',df_acluster1),
										 ('acluster2',df_acluster2),
										 ('acluster3',df_acluster3),
										 ('acluster01',df_acluster01),
										 ('acluster05',df_acluster05)])
acb.iterate()
acb.results.to_csv('acbresults.csv')

colnames = df_ovs_smote1.columns
df_ovs_smote1 = df_ovs_smote1.merge(test,on=['filepath'], how='left')
df_ovs_smote1 = df_ovs_smote1[df_ovs_smote1.cl_y.isnull()].dropna(axis=1)
df_ovs_smote1.columns = colnames
rf_smote1 = randomForestAnalysis(df_ovs_smote1, test)
rf_smote1.iterate(n_estimators_conf=[5,10,25],
		   		  max_depth_conf=[10,20,30])
rf_smote1.results.to_csv('rfsmote1results.csv')
print rf_smote1.results

colnames = df_ovs_smote05.columns
df_ovs_smote05 = df_ovs_smote05.merge(test,on=['filepath'], how='left')
df_ovs_smote05 = df_ovs_smote05[df_ovs_smote05.cl_y.isnull()].dropna(axis=1)
df_ovs_smote05.columns = colnames
rf_smote05 = randomForestAnalysis(df_ovs_smote05, test)
rf_smote05.iterate(n_estimators_conf=[5,10,25],
		   		  max_depth_conf=[10,20,30])
rf_smote05.results.to_csv('rfsmote05results.csv')
print rf_smote05.results

colnames = df_ovs_cluster1.columns
df_ovs_cluster1 = df_ovs_cluster1.merge(test,on=['filepath'], how='left')
df_ovs_cluster1 = df_ovs_cluster1[df_ovs_cluster1.cl_y.isnull()].dropna(axis=1)
df_ovs_cluster1.columns = colnames
rf_cluster1 = randomForestAnalysis(df_ovs_cluster1, test)
rf_cluster1.iterate(n_estimators_conf=[5,10,25],
		   		  max_depth_conf=[10,20,30])
rf_cluster1.results.to_csv('rfcluster1results.csv')
print rf_cluster1.results

colnames = df_ovs_cluster05.columns
df_ovs_cluster05 = df_ovs_cluster05.merge(test,on=['filepath'], how='left')
df_ovs_cluster05 = df_ovs_cluster05[df_ovs_cluster05.cl_y.isnull()].dropna(axis=1)
df_ovs_cluster05.columns = colnames
rf_cluster05 = randomForestAnalysis(df_ovs_cluster05, test)
rf_cluster05.iterate(n_estimators_conf=[5,10,25],
		   		  max_depth_conf=[10,20,30])
rf_cluster05.results.to_csv('rfcluster05results.csv')
print rf_cluster05.results

colnames = df_ovs_acluster1.columns
df_ovs_acluster1 = df_ovs_acluster1.merge(test,on=['filepath'], how='left')
df_ovs_acluster1 = df_ovs_acluster1[df_ovs_acluster1.cl_y.isnull()].dropna(axis=1)
df_ovs_acluster1.columns = colnames
rf_acluster1 = randomForestAnalysis(df_ovs_acluster1, test)
rf_acluster1.iterate(n_estimators_conf=[5,10,25],
		   		  max_depth_conf=[10,20,30])
rf_acluster1.results.to_csv('rfacluster1results.csv')
print rf_acluster1.results

colnames = df_ovs_acluster05.columns
df_ovs_acluster05 = df_ovs_acluster05.merge(test,on=['filepath'], how='left')
df_ovs_acluster05 = df_ovs_acluster05[df_ovs_acluster05.cl_y.isnull()].dropna(axis=1)
df_ovs_acluster05.columns = colnames
rf_acluster05 = randomForestAnalysis(df_ovs_acluster05, test)
rf_acluster05.iterate(n_estimators_conf=[5,10,25],
		   		  max_depth_conf=[10,20,30])
rf_acluster05.results.to_csv('rfacluster05results.csv')
print rf_acluster05.results



