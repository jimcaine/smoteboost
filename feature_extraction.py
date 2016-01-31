####################################################################################
# Jim Caine
# March, 2015
# CSC529 (Inclass)
# Final Project - Synthetic Sampling Boost / Plankton
# caine.jim@gmail.com
####################################################################################


import glob
import random
import numpy as np
import scipy as sp
import pandas as pd
from skimage.io import imread
from skimage import transform
from skimage import morphology
from skimage import measure

DATA_PATH = './data/'

def generate_list_of_class_labels():
	class_labels = []
	class_labels0 = glob.glob('./train/*')
	for cl in class_labels0:
		class_labels.append(cl.split('/')[2])
	return class_labels


def get_radial_density_features(im):
	inner_cords = set()
	x0 = (im.shape[0]/2)-1
	y0 = (im.shape[1]/2)-1
	deltax = 2
	deltay = 2
	radial_means = []
	radial_stds = []
	radial_skews = []
	radial_kurtosis = []
	while x0 > 0:
		densities_oneradius = []
		for dx in range(deltax):
			for dy in range(deltay):
				if (x0+dx, y0+dy) not in inner_cords:
					densities_oneradius.append(im[x0+dx][y0+dy])
					inner_cords.add((x0+dx, y0+dy))
		radial_means.append(np.mean(densities_oneradius))
		radial_stds.append(np.std(densities_oneradius))
		radial_skews.append(sp.stats.skew(densities_oneradius))
		radial_kurtosis.append(sp.stats.kurtosis(densities_oneradius))
		x0 -= 2
		y0 -= 2
		deltax += 2
		deltay += 2

	return radial_means + radial_stds + radial_skews + radial_kurtosis


def get_hist_features(im):
	return np.histogram(im,bins=256)[0]


def get_segmentation_features(im):
	dilwindow = [4,4]
	imthr = np.where(im > np.mean(im),0.0,1.0)
	imdil = morphology.dilation(imthr, np.ones(dilwindow))
	labels = measure.label(imdil)
	labels = imthr*labels
	labels = labels.astype(int)
	regions = measure.regionprops(labels)
	numregions = len(regions)
	while len(regions) < 1:
		dilwindow[0] = dilwindow[0] - 1
		dilwindow[1] = dilwindow[1] - 1
		if dilwindow == [0,0]:
			regions = None
			break
		imthr = np.where(im > np.mean(im),0.0,1.0)
		imdil = morphology.dilation(imthr, np.ones(dilwindow))
		labels = measure.label(imdil)
		labels = imthr*labels
		labels = labels.astype(int)
		regions = measure.regionprops(labels)
	regionmax = get_largest_region(regions,labels,imthr)

	if regionmax is None:
		return (np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan)
	eccentricity = regionmax.eccentricity
	convex_area = regionmax.convex_area
	convex_to_total_area = regionmax.convex_area / regionmax.area
	extent = regionmax.extent
	filled_area = regionmax.filled_area
	return (eccentricity, convex_area, convex_to_total_area,
			extent, filled_area, numregions)


def get_largest_region(regions, labelmap, imagethres):
	if regions is None:
		return None
	else:
	    regionmaxprop = None
	    for regionprop in regions:
	        # check to see if the region is at least 50% nonzero
	        if sum(imagethres[labelmap == regionprop.label])*1.0/regionprop.area < 0.50:
	            continue
	        if regionmaxprop is None:
	            regionmaxprop = regionprop
	        if regionmaxprop.filled_area < regionprop.filled_area:
	            regionmaxprop = regionprop
	    return regionmaxprop


def generate_dataset(trainortest='train'):
	# generate df with filename (and class label for test set)
	if trainortest == 'train':
		files = []
		class_labels = generate_list_of_class_labels()
		for cl in class_labels:
			filepaths = glob.glob('./train/%s/*' % cl)
			for f in filepaths:
				files.append((cl, f))
		df = pd.DataFrame(files)
		df.columns = ['cl', 'filepath']
		# df = df.loc[0:1000]
	elif trainortest == 'test':
		filepaths = glob.glob('./test/*')
		df = pd.DataFrame(filepaths)
		df.columns = ['filepath']

	# keep a counter to track progress
	counter = 0

	# iterate through each row (image) and save features to list
	features_allimages = []
	for index, value in df.iterrows():
		features_oneimage = []
		# keep track of progress
		counter += 1
		if counter % 250 == 0:
			print counter

		# load the image and create derivative images
		im = imread(value['filepath'])


		# # add histogram features for one image
		# hist_feat = get_hist_features(im)
		# for f in hist_feat:
		# 	features_oneimage.append(f)

		# add radial density features for one image
		imsizes = [(16,16)]
		for size in imsizes:
			imres = transform.resize(im, size)
			rdf = get_radial_density_features(imres)
			for f in rdf:
				features_oneimage.append(f)

		# add segmentation features
		segm_feat = get_segmentation_features(im)
		for f in segm_feat:
			features_oneimage.append(f)

		# add this images features to features_allimages list
		features_allimages.append(features_oneimage)

	# concat the new features to the existing dataframe
	df_features = pd.DataFrame(features_allimages)
	df = pd.concat([df, df_features], axis=1)

	# return the df
	df.columns = ['cl','filepath',
					   'rd1',
					   'rd2',
					   'rd3',
					   'rd4',
					   'rd5',
					   'rd6',
					   'rd7',
					   'rd8',
					   'rd9',
					   'rd10',
					   'rd11',
					   'rd12',
					   'rd13',
					   'rd14',
					   'rd15',
					   'rd16',
					   'eccentricity',
					   'convex_area',
					   'convex_to_total_area',
					   'extent',
					   'filled_area',
					   'numregions']
	return df




df = generate_dataset(trainortest='train')
# sample_ind = random.sample(df, 5000)
# df = df.iloc[sample_ind]
df.to_csv(DATA_PATH + 'dataset.csv')
print df