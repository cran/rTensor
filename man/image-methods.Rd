\docType{methods}
\name{image-methods}
\alias{image,Tensor-method}
\alias{image-methods}
\title{Image of 3-Tensor Slices}
\arguments{
  \item{x}{the Tensor instance}

  \item{m}{the mode to slice along}

  \item{indices}{the range of indices to be plotted, by
  default all indices along mode \code{m}}
}
\description{
  Calls \code{image} on the multiple slices of the
  3-Tensors to provide more conveinence than calling
  \code{image} on \code{tnsr@data}. Need to specify which
  mode and the range of index along that mode.
}
\details{
  \code{image(x,m=NULL,indices=NULL,...)}
}
\examples{
tnsr <- rand_tensor(c(3,4,5))
#image(tnsr,m=3,indices=c(2:4))
#image(tnsr,m=2)
}

