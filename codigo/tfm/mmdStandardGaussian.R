##
# Function that computes the kernel width and the kernel matrix
#

calculateKernel <- function(sample, r) {
	
	computeDistance <- function(sample) {

		n <- length(sample)
    if (n == 1) return(0)
    
		sample <- matrix(sample, n, 1)
		Q <- matrix(sample^2, n, n)
		R <- t(Q)
		ret <- Q + R - 2 * sample %*% t(sample)
		ret[ upper.tri(ret) ]
	}

	n <- length(sample)

	sample <- matrix(sample, n, 1)

	Q <- matrix(sample^2, n, n)
	R <- t(Q)
	distance <- Q + R - 2 * sample %*% t(sample)

	if (r > 1)
		sigma <- sqrt(0.5 * (median(distance[ upper.tri(distance) ])))
	else
		sigma <- sqrt(0.5 * (median(computeDistance(sample[ sample > 0 ]))))

	K <- exp(-0.5 * distance / sigma^2)

	list(K = K, sigma = sigma)
}

##
# Computation of the mmd of a sample to a standard Gaussian distribution
# and standard deviation of the statistic.
#
 
mmdStandardGaussian <- function(sample, r) {

  #sample <- as.array(sample) #Añadido para que el sum de una matriz sea un número
	n <- length(sample)

	# We compute the kernel width and the kenel matrix

	ret <- calculateKernel(sample, r)
	K <- ret$K
	sigma <- ret$sigma

	nonRandomTerm <- sigma / sqrt(2 + sigma^2)

	expectationKernelStandardGaussian <- function(x) sqrt(sigma^2 / (1 + sigma^2)) * exp(-0.5 * x^2 / (1 + sigma^2))
	Y <- matrix(expectationKernelStandardGaussian(sample), n, n)
 
	diag(Y) <- 0
	diag(K) <- 0

	randomTerm <- 1 / (n * (n - 1)) * base::sum(K - Y - t(Y))

	# We compute the mmd statistic
  MMD <- randomTerm + nonRandomTerm

	# We compute the standard deviation of the mmd
	# statistic using the formula from Sefling's method

	expectationX2 <- 1 / (n - 1) * apply(K - Y - t(Y), 2, sum)
	sdMMD <- sqrt(4 / n * (mean(expectationX2^2) - mean(expectationX2)^2))

	list(MMD = MMD, sdMMD = sdMMD)
}

##
# Compute MMD decision.
#
# This function returns a number between 0 and 1. When this number
# is larger than 0.5, the first sample is estimated to be closer to
# a Gaussian distribution. When that number is lower than 0.5,
# the second sample is estimated to be closer to a Gaussian distribution.
#

computeMMDdecision <- function(sample1, sample2, r) {

	sample1 <- (sample1 - mean(sample1)) / sd(sample1)
	sample2 <- (sample2 - mean(sample2)) / sd(sample2)

	ret1 <- mmdStandardGaussian(sample1, r)
	ret2 <- mmdStandardGaussian(sample2, r)

	integrate(function(x) dnorm(x, ret1$MMD, ret1$sdMMD) * pnorm(x, ret2$MMD, ret2$sdMMD),
		min(ret1$MMD, ret2$MMD) - 6 * max(ret1$sdMMD, ret2$sdMMD),
		max(ret1$MMD, ret2$MMD) + 6 * max(ret1$sdMMD, ret2$sdMMD), subdivisions = 1e+07)$value
}
