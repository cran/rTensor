\docType{methods}
\name{[-methods}
\alias{[,Tensor-method}
\alias{[-methods}
\alias{extract,Tensor-method}
\title{Extract for Tensor}
\arguments{
  \item{x}{Tensor to be subset}

  \item{i,j,...}{indices that specify the extents of the
  sub-tensor}

  \item{drop}{whether or not to reduce the number of modes
  to exclude those that have '1' as the mode}
}
\value{
  subTensor of class Tensor
}
\description{
  Extends '[' for Tensor class. Works exactly as it would
  for the base 'array' class.
}
\details{
  \code{x[i,j,...,drop=TRUE]}
}
\examples{
tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
tnsr[1,2,3]
tnsr[3,1,]
tnsr[,,5]
tnsr[,,5,drop=FALSE]
}

