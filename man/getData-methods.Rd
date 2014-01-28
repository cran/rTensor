\docType{methods}
\name{getData-methods}
\alias{getData}
\alias{getData,Tensor-method}
\alias{getData-methods}
\title{Data Getter for Tensor}
\arguments{
  \item{tnsr}{the Tensor instance}
}
\value{
  a (\code{array}, \code{matrix}, or \code{vector}) that
  holds the actual data
}
\description{
  Return the data (\code{array}, \code{matrix}, or
  \code{vector}) from a tensor. Safer than accessing the
  slot directly.
}
\details{
  \code{getData(tnsr)}
}
\examples{
tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
getData(tnsr)
}

