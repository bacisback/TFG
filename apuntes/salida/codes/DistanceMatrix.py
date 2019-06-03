size = len(x)
G = np.sum(x*x,1) #Here we calculate the square of each sample
Q = np.repeat(G,size).reshape(size,size) #row i contain each the square of sample i n times
R = Q.T #colum i contain each the square of sample i n times
dists = Q + R -2*np.dot(xmed,xmed.T) #we calculate (x - y)^2 = x^2+y^2-2x*y
dists = dists - np.tril(dists) #we remove repeated distances (x-y)^2 = (y-x)^2
dists = dists.reshape(size*size,1)
hyperparameter = np.sqrt(0.5*np.median(dists)) #Calculate the hyperparameter of our kernel