\docType{methods}
\name{dim-methods}
\alias{dim,Tensor-method}
\alias{dim-methods}
\title{Mode Getter for Tensor}
\arguments{
  \item{tnsr}{the Tensor instance}
}
\value{
  an integer vector of the modes associated with \code{x}
}
\description{
  Return the vector of modes from a tensor. Safer than
  accessing the slot directly.
}
\details{
  \code{dim(tnsr)}
}
\examples{
tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
dim(tnsr)
}
\seealso{
  \code{\link{getModes}}
}

