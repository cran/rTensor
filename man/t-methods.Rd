\docType{methods}
\name{t-methods}
\alias{t,Tensor-method}
\alias{t-methods}
\title{Tensor Transpose}
\arguments{
  \item{tnsr}{a 3-tensor}
}
\value{
  tensor transpose of \code{x}
}
\description{
  Implements the tensor transpose based on block circulant
  matrices (Kilmer et al. 2013) for 3-tensors.
}
\details{
  \code{t(tnsr)}
}
\examples{
tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
identical(t(tnsr)@data[,,1],t(tnsr@data[,,1]))
identical(t(tnsr)@data[,,2],t(tnsr@data[,,5]))
identical(t(t(tnsr)),tnsr)
}
\references{
  M. Kilmer, K. Braman, N. Hao, and R. Hoover, "Third-order
  tensors as operators on matrices: a theoretical and
  computational framework with applications in imaging".
  SIAM Journal on Matrix Analysis and Applications 2013.
}

