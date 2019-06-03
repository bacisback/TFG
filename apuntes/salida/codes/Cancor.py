def cancor(x,y,k):
	canonical_correlation_matrix = np.cov(np.hstack([x, y]).T)

	k0 = k
	lower_bound = 1 #minimum k
	upper_bound = k #maximum k
	while True:
		#Canonical correlation
		k = int(k)

		C_XX = canonical_correlation_matrix[:k,:k]
		C_YY = canonical_correlation_matrix[k0:k0+k, k0:k0+k]
		C_XY = canonical_correlation_matrix[:k, k0:k0+k]
		C_YX = canonical_correlation_matrix[k0:k0+k, :k]

		eigs = np.linalg.eigvals(np.dot(np.dot(np.linalg.inv(C_XX), C_XY),
										np.dot(np.linalg.inv(C_YY), C_YX)))

		#Search if K is too large
		if not (np.all(np.isreal(eigs)) and
				0 <= np.min(eigs) and
				np.max(eigs) <= 1): #Condition of being too large
			upper_bound -= 1 #reduce the maximum in 1
			k = (upper_bound + lower_bound) / 2 #search in the middle 
			continue

		if lower_bound == upper_bound: break #if lower_bound == upper_bound means we found the optimal value for k

		lower_bound = k #as k meets the condition we set the lower bound to k

		#Set k as the middle point
		if upper_bound == lower_bound + 1: 
			k = upper_bound

		else:
			k = (upper_bound + lower_bound) / 2

	return np.sqrt(eigs),k