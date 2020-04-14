
cimport cython
from libc.math cimport sqrt, log

@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)

cpdef find_cluster_s1(float[::1] point, float[:, ::1] centres, int[::1] counts, ):
	# calculate posterior probability for stage 1 
	cdef Py_ssize_t N = point.shape[0]
	cdef Py_ssize_t M = centres.shape[0]
	cdef Py_ssize_t i, j
	cdef float dist, prob, max_prob = -10000
	cdef int cluster_label = 0


	for i in range(M):
		dist = 0.0
		for j in range(N):
			dist += (point[j] - centres[i, j]) * (point[j] - centres[i, j])
		dist = sqrt(dist)
		prob = log(counts[i]) - dist

		if prob > max_prob:
			max_prob = prob
			cluster_label = i

	return cluster_label, max_prob


@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)

cpdef int find_cluster_s2(float[::1]point, float[:, ::1]centres, int[::1]counts, float[::1] likelihood):
	# calculate posterior probability for stage 2
	cdef Py_ssize_t N = point.shape[0]
	cdef Py_ssize_t M = centres.shape[0]
	cdef Py_ssize_t i, j
	cdef float dist, prob, max_prob = -10000
	cdef int cluster_label = 0

	for i in range(M):
		dist = 0.0
		for j in range(N):
			dist += (point[j] - centres[i,j]) * (point[j] - centres[i,j]) 
		dist = sqrt(dist)

		prob = log(counts[i]) - dist + likelihood[i]

		if prob > max_prob:
			max_prob = prob
			cluster_label = i

	return cluster_label



@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef update_centre(float[::1] point, float[::1] centre, int count, int sign):
	# update centre location 
	cdef Py_ssize_t N = point.shape[0]
	cdef Py_ssize_t i = 0

	for i in range(N):
		centre[i] = centre[i] + sign * (point[i] - centre[i]) / count

	return centre

