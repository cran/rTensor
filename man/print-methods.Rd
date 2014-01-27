\docType{methods}
\name{print-methods}
\alias{print,Tensor-method}
\alias{print-methods}
\title{Print for Tensor}
\arguments{
  \item{x}{the Tensor instance}

  \item{...}{additional parameters to be passed into
  print()}
}
\description{
  Extend print for Tensor
}
\details{
  \code{print(x,...)}
}
\examples{
tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
print(tnsr)
}
\seealso{
  \code{\link{show}}
}

