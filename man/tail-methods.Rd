\docType{methods}
\name{tail-methods}
\alias{tail,Tensor-method}
\alias{tail-methods}
\title{Tail for Tensor}
\arguments{
  \item{x}{the Tensor instance}

  \item{...}{additional parameters to be passed into
  tail()}
}
\description{
  Extend tail for Tensor
}
\details{
  \code{tail(x,...)}
}
\examples{
tnsr <- rand_tensor()
tail(tnsr)
}
\seealso{
  \code{\link{head-methods}}
}

