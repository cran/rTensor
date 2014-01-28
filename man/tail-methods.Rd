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
tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
tail(tnsr)
}
\seealso{
  \code{\link{head-methods}}
}

