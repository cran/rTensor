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
  Return the vector of modes from a tensor
}
\details{
  \code{dim(tnsr)}
}
\examples{
tnsr <- rand_tensor()
dim(tnsr)
}

