\docType{methods}
\name{head-methods}
\alias{head,Tensor-method}
\alias{head-methods}
\title{Head for Tensor}
\arguments{
  \item{x}{the Tensor instance}

  \item{...}{additional parameters to be passed into
  head()}
}
\description{
  Extend head for Tensor
}
\details{
  \code{head(x,...)}
}
\examples{
tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
head(tnsr)
}
\seealso{
  \code{\link{tail-methods}}
}

