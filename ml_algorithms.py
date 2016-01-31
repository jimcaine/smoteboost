####################################################################################
# Jim Caine
# March, 2015
# CSC529 (Inclass)
# Final Project - Synthetic Sampling Boost / Plankton
# caine.jim@gmail.com
####################################################################################


import random
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as DT

class smoteBoost(object):
	def __init__(self, base_learner=DT(max_depth=2),
				 n_estimators=3,
				 smote_ratio=10,
				 class_numsamples_dict=False,
				 df_smote=False,
				 smote_decay='linear'):
		self.m = base_learner
		self.T = n_estimators
		self.smote_ratio = smote_ratio
		self.class_numsamples_dict = class_numsamples_dict
		self.df_smote = df_smote
		self.smote_decay = smote_decay

	def fit(self, x, y):
		# initialize list to hold models and model weights (alphas)
		self.models = []
		self.alphas = []

		xconstant = x.copy()
		yconstant = y.copy()

		# initialize weights
		w0 = 1.0 / x.shape[0]
		w = pd.Series([w0]*x.shape[0], index=x.index)
		w.name = 'w'
		w_index = w.index.values

		# iterate T times
		for i in range(self.T):
			# modify the distribution by creating N synthetic examples from minority class
			synthetic_indices = []
			for c in self.class_numsamples_dict:
				df_c = self.df_smote[self.df_smote.cl == c]
				synthetic_ind = random.sample(df_c.index, \
					np.minimum(int(self.class_numsamples_dict[c]*self.smote_ratio), df_c.shape[0]))
				for i in synthetic_ind:
					synthetic_indices.append(i)
			synthetic_df_round = self.df_smote.loc[synthetic_indices]
			xsynthetic = synthetic_df_round.drop(['cl','filepath'], axis=1)
			ysynthetic = synthetic_df_round.cl
			xsmote = pd.concat([x, xsynthetic])
			ysmote = pd.concat([y, ysynthetic])

			# train a weak learner
			clf = self.m.fit(xsmote, ysmote)
			self.models.append(clf)

			# make predictions and compute loss
			predtrain = clf.predict(xconstant)
			df_err = pd.DataFrame({'pred':predtrain, 'actual':y})
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
										   size=xconstant.shape[0],
										   replace=True,
										   p=w.values)
			x = xconstant.loc[new_indices]
			y = yconstant.loc[new_indices]	

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
				rus_ind = random.sample(df_c.index, np.minimum(10, df_c.shape[0]))
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



class dataBoostIM(object):
	def __init__(self, base_learner=DT(max_depth=2),
				 num_iterations=3):
		self.m = base_learner
		self.T = num_iterations

	def fit(self, x, y):
		# initialize list to hold models and model weights (alphas)
		self.models = []
		self.betas = []

		# create copies of x and y to retrieve after mutation
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
				# identify hard examples from the original data set for different classes
				new_indices = np.random.choice(w.index,
											   size=xconstant.shape[0],
											   replace=True,
											   p=w.values)
				xsynthetic = xconstant.loc[new_indices]
				ysynthetic = yconstant.loc[new_indices]

				# generate synthetic data to balance the training knowledge of different classes

				# add synthetic data to the original training set to form a new training data set

				# update and balance the total weights of the different classes in the new
				# training set

			# train a weak learner
			clf = self.m.fit(xsynthetic, ysynthetic)
			self.models.append(clf)

			# get back the hypothesis
			pred = clf.predict(xconstant)

			# calculate the error
			df_err = pd.DataFrame({'pred':pred, 'actual':y})
			h_t = (df_err['pred'] != df_err['actual'])*1
			e = h_t.values.dot(w.values).sum()

			# calculate beta
			beta = float(e)/(1-e)
			self.betas.append(beta)

			# update weights
			h = h_t.replace(1,beta).replace(0,1).values
			w = w * h
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
		# predict the class with the largest sum of log betas
		weighted_predictions = []
		for index, value in predictions.iterrows():
			predictions_prob_dict = {}
			for i in range(len(self.betas)):
				v = value.iloc[i]
				if v not in predictions_prob_dict.keys():
					predictions_prob_dict[v] = np.log(self.betas[i])
				else:
					predictions_prob_dict[v] += np.log(self.betas[i])
			weighted_predictions.append(max(predictions_prob_dict, key=predictions_prob_dict.get))
		return weighted_predictions

	def predict_proba(self, x):
		predictions = []
		for m in self.models:
			pred_proba = m.predict_proba(x)
			predictions.append(pred_proba)
		weighted_predictions = []
		for i in range(len(predictions)):
			weighted_predictions.append(self.betas[i]*predictions[i])
		predict_proba = sum(weighted_predictions)
		return predict_proba



class ABM2(object):
	def __init__(self,
				 base_learner=DT(max_depth=2),
				 n_estimators=3,
				 synthetic_data=None,
				 synthetic_ratio=1.0):
		self.m = base_learner
		self.T = n_estimators
		self.df_synthetic = synthetic_data
		self.synthetic_ratio = synthetic_ratio

	def fit(self, x, y):
		# calculate the number of samples per class that need to be generated
		if self.df_synthetic != None:
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
				# identify hard examples from the original data set for different classes
				new_indices = np.random.choice(w.index,
											   size=xconstant.shape[0],
											   replace=True,
											   p=w.values)
				xsynthetic = xconstant.loc[new_indices]
				ysynthetic = yconstant.loc[new_indices]

				# ensure that every class is represented in new dataframe
				if len(ysynthetic.unique()) != 121:
					non_represented_classes = np.setdiff1d(class_list, ysynthetic.unique())
					for c in non_represented_classes:
						index = random.sample(yconstant[yconstant == c].index, 1)
						xunderrep = xconstant.loc[index]
						yunderrep = yconstant.loc[index]
						xsynthetic = pd.concat([xsynthetic, xunderrep], axis=0)
						ysynthetic = pd.concat([ysynthetic, yunderrep])


				if self.df_synthetic != None:
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

			# train a weak learner
			clf = self.m.fit(xsynthetic, ysynthetic)
			self.models.append(clf)

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