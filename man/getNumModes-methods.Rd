\docType{methods}
\name{getNumModes-methods}
\alias{getNumModes}
\alias{getNumModes,Tensor-method}
\alias{getNumModes-methods}
\title{Number of Modes Getter for Tensor}
\arguments{
  \item{tnsr}{the Tensor instance}
}
\value{
  an integer of the number of modes associated with
  \code{x}
}
\description{
  Return the number of modes from a tensor. Safer than
  accessing the slot directly.
}
\details{
  \code{getNumModes(tnsr)}
}
\examples{
tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
getNumModes(tnsr)
}

