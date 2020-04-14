# using Gibbs sampling
import numpy as np
from utils import find_cluster_s1, find_cluster_s2, update_centre

class clustered_object(object):
	def __init__(self, clustered_set=None, cluster_labels=None, cluster_counts=None, cluster_centres=None):
		self.clustered_set = clustered_set
		self.cluster_labels = cluster_labels
		self.cluster_counts = cluster_counts
		self.cluster_centres = cluster_centres


class modified_dpmm(object):
	def __init__(self, dataset, beta, r, max_cluster):   
		self.dataset = dataset
		self.beta = beta          # initial radius
		self.r = r                # learning rate
		self.MAX_CLUSTERS = max_cluster   # max possible number clusters in a set
		self.N = self.dataset.shape[1] # number of features in one data point
		self.M = self.dataset.shape[0] # number of points in the dataset
		self.indices = None


	#@profile
	def find_initial_labels(self):  
		# stage 1 - find initial number of clusters
		K = 0 
		d_copy = self.dataset.view() # create view of dataset
		N = self.N
		M = self.M
		beta = self.beta
		centres = np.zeros((self.MAX_CLUSTERS, N-2), dtype = np.float32) # temporary variable for centres
		counts = np.ones((self.MAX_CLUSTERS), dtype=np.int32) # temporary variable for counts
		res = clustered_object()

		# add first point to first cluster
		centres[K] += (d_copy[0, 1:N-1] - centres[K]) / counts[K]
		counts[K] += 1
		d_copy[0, N-1] = K
		K += 1

		for i in range(1, M):
			
			data_tmp = d_copy[i, 1:N-1]

			(best_key, max_prob) = find_cluster_s1(data_tmp, centres[0:K], counts[0:K],) #find max probability and return corresponding index 

			if max_prob < -beta:
				best_key = K
				K += 1

			centre_tmp = centres[best_key]
			new_centre = update_centre(data_tmp, centre_tmp, counts[best_key], sign = 1)
			centres[best_key] = new_centre
			counts[best_key] = counts[best_key] + 1
			
			d_copy[i, N-1] = best_key
			

		res.clustered_set = self.dataset
		res.cluster_labels, res.cluster_counts = np.unique(d_copy[:,N-1], return_counts = True)
		res.cluster_centres = centres[0:K]
		return res


	#@profile
	def assign_labels(self, label_ids=None, prev_counts=None, prev_centres=None, converge=False, numIter=10):
		if label_ids is None:
			# no previously assigned labels -> stage 1
			return self.find_initial_labels()

		else:
			if not converge:
				# to run just one iteration of stage 2 (Academic purposes only) 
				K = len(label_ids)
				centres = prev_centres
				new_counts = np.ones((K,), dtype=np.int32) # temporary variable for counts
				d_copy = self.dataset.view() # create view of dataset
				N = self.N
				M = self.M
				likelihood = (np.log(prev_counts) * self.r).astype(np.float32)
				res = clustered_object()

				for i in range(M):
					data_tmp = d_copy[i, 1:N-1]

					(best_key) = find_cluster_s2(data_tmp, centres, new_counts, likelihood) #find max probability and return corresponding index 
					
					centre_tmp = centres[best_key]
					new_centre = update_centre(data_tmp, centre_tmp, new_counts[best_key], sign = 1)
					centres[best_key] = new_centre
					new_counts[best_key] = new_counts[best_key] + 1
			
					d_copy[i, N-1] = label_ids[best_key]


				active_idx = new_counts != 1 # get mask for non zero clusters
				# remove counts and centriods of empty or inactive clusters
				res.clustered_set = self.dataset
				res.cluster_labels, self.indices =  np.unique(d_copy[:, N-1], return_inverse = True)
				res.cluster_counts = new_counts[active_idx]
				res.cluster_centres = centres[active_idx]
				return res

			else:
				# stage 2 - iterate till convergence
				d_copy = self.dataset.view()
				N = self.N
				M = self.M
				
				counts = prev_counts
				cur_centres = prev_centres.copy()
				labels = d_copy[:,N-1]
				res = clustered_object()

				for n in range(numIter):
					likelihood = (np.log(counts) * self.r).astype(np.float32)
					
					for i in range(M):
						data_tmp = d_copy[i, 1:N-1]
						old_idx = self.indices[i] #idx[labels[i]]

						# unassign from the previously assigned cluster
						if counts[old_idx] >=1:
							centre_tmp = cur_centres[old_idx]
							new_centre = update_centre(data_tmp, centre_tmp, counts[old_idx], sign = -1)
							cur_centres[old_idx] = new_centre
							counts[old_idx] -= 1						

						# calculate posterior probability
						(best_key) = find_cluster_s2(data_tmp, cur_centres, counts, likelihood) #find max probability and return corresponding index 

						# reassign to correct cluster
						centre_tmp = cur_centres[best_key]
						new_centre = update_centre(data_tmp, centre_tmp, counts[best_key], sign = 1)
						cur_centres[best_key] = new_centre
						counts[best_key] += 1
						# d_copy[i, N-1] = label_ids[best_key]
						labels[i] = label_ids[best_key]


					# remove inactive clusters
					active_idx = counts > 1
					label_ids, counts = label_ids[active_idx], counts[active_idx]
					l_tmp, self.indices = np.unique(d_copy[:, N-1], return_inverse = True)
					cur_centres = cur_centres[active_idx]


					if np.array_equal(cur_centres, prev_centres):
						# if centroids are not changing anymore, return after removing inactive clusters
						active_idx = counts > 1
						print('number of iterations = %d' % n)
						res.clustered_set = self.dataset
						res.cluster_labels, res.cluster_counts = label_ids[active_idx], counts[active_idx]
						res.cluster_centres = cur_centres[active_idx]
						return res

					else:
						prev_centres = cur_centres.copy()

				res.clustered_set = self.dataset
				res.cluster_labels = label_ids.astype(int)
				res.cluster_counts = counts
				res.cluster_centres = cur_centres

				return res

				
				