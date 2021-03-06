\name{pdcor}
\alias{pdcor}
\alias{pdcov}
\alias{pdcor.test}
\alias{pdcov.test}
\title{
Partial distance correlation and covariance
}
\description{Partial distance correlation pdcor, pdcov, and tests.}
\usage{
  pdcov.test(x, y, z, R)
  pdcor.test(x, y, z, R)
  pdcor(x, y, z)
  pdcov(x, y, z)
}
\arguments{
\item{x}{ data matrix or dist object of first sample}
\item{y}{ data matrix or dist object of second sample}
\item{z}{ data matrix or dist object of third sample}
\item{R}{ replicates for permutation test}
}
\details{
\code{pdcor(x, y, z)} and \code{pdcov(x, y, z)} compute the partial distance
correlation and partial distance covariance, respectively,
of x and y removing z.

A test for zero partial distance correlation (or zero partial distance covariance)
is implemented in \code{pdcor.test}, and \code{pdcov.test}.

If the argument is a matrix, it is treated as a data matrix and distances
are computed (observations in rows). If the arguments are distances or
dissimilarities, they must be distance (\code{dist}) objects. For symmetric,
zero-diagonal dissimilarity matrices, use \code{as.dist} to convert to a
\code{dist} object.
}
\value{
Each test returns an object of class \code{htest}.
}
\author{ Maria L. Rizzo \email{mrizzo @ bgsu.edu} and
Gabor J. Szekely
}
\references{
Szekely, G.J. and Rizzo, M.L. (2014),
 Partial Distance Correlation with Methods for Dissimilarities.
 \emph{Annals of Statistics}, Vol. 42 No. 6, 2382-2412.
}
\examples{
  n = 30
  R <- 199

  ## mutually independent standard normal vectors
  x <- rnorm(n)
  y <- rnorm(n)
  z <- rnorm(n)

  pdcor(x, y, z)
  pdcov(x, y, z)
  pdcov.test(x, y, z, R=R)
  print(pdcor.test(x, y, z, R=R))


if (require(MASS)) {
  p = 4
  mu <- rep(0, p)
  Sigma <- diag(p)

  ## linear dependence
  y <- mvrnorm(n, mu, Sigma) + x
  print(pdcov.test(x, y, z, R=R))

  ## non-linear dependence
  y <- mvrnorm(n, mu, Sigma) * x
  print(pdcov.test(x, y, z, R=R))
  }
}
\keyword{ htest }
\keyword{ multivariate }
\keyword{ nonparametric }
\concept{ independence }
\concept{ multivariate }
\concept{ distance correlation }
\concept{ distance covariance }
\concept{ energy statistics }
