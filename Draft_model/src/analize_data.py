import numpy as np
from os.path import isfile, join
from tqdm import tqdm
from scipy.interpolate import BSpline, splrep 
import matplotlib.pyplot as plt
import time
import random as rd

# check if the excluded_data.npy file exists, if it does, load it, if it does not,
# create it
def get_excluded_data(file="excluded_data.npy"):
	'''
	This function reads the raw file (named data) and the clean file (named
	data_clean) and returns the data that is in the raw file but not in the clean
	file. The data is saved in a .npy file named "excluded_data.npy". If the file
	already exists, the function reads the data from the file instead of creating
	it again.
	'''
	dir = "../data"
	if isfile(join(dir, file)):
		excluded_data = np.load(join(dir, file), allow_pickle=True)
	else:
		# Load the data
		data = np.load(join(dir, "data.npy"), allow_pickle=True)
		data_clean = np.load(join(dir,"data_clean.npy"), allow_pickle=True)

		# get the labels that are not in the clean data
		label_list = [label for label in tqdm(data['label']) if label not in data_clean['label']]

		# get the data that is not in the clean data
		excluded_data = np.array([data[data['label'] == label] for label in label_list])

		# save the excluded data in a .npy file named "excluded_data.npy"
		np.save(join(dir, file), excluded_data)
		print("Excluded data saved in excluded_data.npy")
	return excluded_data

def get_mean_distance(x, y, yerr):
	'''
	This function gets the x and y data (meaning the coordinates and the 
	brilliances) from the .npy file and computes the mean distance between the
	data points and the spline that is fitted to the data. 
	'''	
	# Divide the data between the horizontal and vertical coordinates
	mask_hor = y[:49] > 0	# mask for the horizontal coordinates
	mask_ver = y[49:] > 0	# mask for the vertical coordinates

	x_hor = x[:49][mask_hor]  # usable horizontal coordinates
	y_hor = y[:49][mask_hor]  # usable horizontal brilliances
	yerr_hor = yerr[:49][mask_hor]  # usable horizontal errors
	y_hor_index = np.arange(49)[mask_hor]  # indices of the usable horizontal coordinates

	x_ver = x[49:][mask_ver]  # usable vertical coordinates
	y_ver = y[49:][mask_ver]  # usable vertical brilliances
	yerr_ver = yerr[49:][mask_ver]  # usable vertical errors
	y_ver_index = np.arange(43)[mask_ver]  # indices of the usable vertical coordinates

	# get a random number between 0 and 1
	rand = rd.random()
	if rand < 0.:
		fig, axs = plt.subplots(1, 2, figsize=(12, 6))
		axs[0].plot(x_hor, y_hor, marker='o', linestyle='--')
		axs[0].set_title('Vertical Coordinates')
		axs[1].plot(x_ver, y_ver, marker='o', linestyle='--')
		axs[1].set_title('Horizontal Coordinates')
		plt.savefig('prova.png')
		plt.close()

	distance_hor_LOS = np.zeros(49)
	distance_ver_LOS = np.zeros(43)
	new_yerr_hor = np.zeros(49)
	new_yerr_ver = np.zeros(43)

	# check if it is possible to compute the spline, if not, return np.nan
	if len(x_hor) <= 3 or len(x_ver) <= 3:
		return np.nan, 0, 0, 0, 0 # return 0 if there are not enough data points
	
	# Compute the mean distance for the horizontal and vertical coordinates
	# Horizontal
	# Compute the spline
	tck = splrep(np.sort(x_hor), y_hor[np.argsort(x_hor)], w=1/yerr_hor[np.argsort(x_hor)], k=3, s=25)
	# Compute the reconstructed value
	y_hor_reconstructed = BSpline(*tck)
	distance_hor = np.abs(y_hor - y_hor_reconstructed(x_hor))
	distance_hor_LOS[y_hor_index] = np.abs(y_hor - y_hor_reconstructed(x_hor))
	new_yerr_hor[y_hor_index] = yerr_hor

	# Vertical
	# compute the spline
	tck = splrep(np.sort(x_ver), y_ver[np.argsort(x_ver)], w=1/yerr_ver[np.argsort(x_ver)], k=3, s=25)
	# Compute the reconstructed value
	y_ver_reconstructed = BSpline(*tck)
	distance_ver = np.abs(y_ver - y_ver_reconstructed(x_ver))
	distance_ver_LOS[y_ver_index] = np.abs(y_ver - y_ver_reconstructed(x_ver))
	new_yerr_ver[y_ver_index] = yerr_ver

	# Compute and return the mean distance
	return np.mean(np.concatenate((distance_hor, distance_ver))), distance_hor_LOS, distance_ver_LOS, new_yerr_hor, new_yerr_ver

def main():
	# get the excluded data
	excluded_data = get_excluded_data(file="data_clean.npy").flatten()
	# print the number of profiles
	print(f"Number of profiles: {len(excluded_data)}")

	# initialize the mean distance list
	tmp_mean_distance = []
	tmp_hor = np.zeros(49)
	tmp_ver = np.zeros(43)
	sum_err_hor = np.zeros(49)
	sum_err_ver = np.zeros(43)
	count_hor = np.zeros(49)
	count_ver = np.zeros(43)

	# Declare a counter for the profiles with not enough data points
	counter = 0

	for prel_value, data_value, err_value in zip(excluded_data['prel'], excluded_data['data'], excluded_data['data_err']):
		# get the mean distance, the data arrays and the errors
		md, hor, vert, err_hor, err_ver = get_mean_distance(prel_value, data_value, err_value)
		
		if md is np.nan:
			counter += 1
		else:
			tmp_mean_distance.append(md)
			# count the non zero occurrences in hor and vert
			x_ind = np.where(hor != 0)
			y_ind = np.where(vert != 0)
			# add the values to the tmp_hor and tmp_ver
			# check if there are any nan in hor or vert and in that case rewrite the arrays with zeros
			if np.isnan(hor).any():
				hor = np.zeros(49)
			if np.isnan(vert).any():
				vert = np.zeros(43)
			# if np.isnan(err_hor).any():
			# 	err_hor = np.zeros(49)
			# if np.isnan(err_ver).any():
			# 	err_ver = np.zeros(43)
			# add the values to the tmp_hor and tmp_ver arrays (these are the distances)
			tmp_hor += hor
			tmp_ver += vert
			# add the errors to the err_hor and err_ver arrays 
			sum_err_hor += err_hor
			sum_err_ver += err_ver
			# count the number of occurrences
			count_hor[x_ind] += 1
			count_ver[y_ind] += 1
			
	# print the mean distance
	print(f"Mean distance: {np.nanmean(tmp_mean_distance)}")
	print(f"Number of profiles with not enough data points: {counter}")
	print(f"Mean distance for the horizontal coordinates: {(tmp_hor/count_hor)}")
	print(f"Mean distance for the vertical coordinates: {(tmp_ver/count_ver)}")
	print(f"Mean error for the horizontal coordinates: {(sum_err_hor/count_hor)}")
	print(f"Mean error for the vertical coordinates: {(sum_err_ver/count_ver)}")
	print(f"Number of profiles with not enough data points: {counter}")

if __name__ == "__main__":
    main()