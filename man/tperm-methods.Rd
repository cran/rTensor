\docType{methods}
\name{tperm-methods}
\alias{tperm}
\alias{tperm,Tensor-method}
\alias{tperm-methods}
\title{Mode Permutation for Tensor}
\arguments{
  \item{x}{the Tensor instance}

  \item{perm}{the new permutation of the current modes}
}
\description{
  Overloads \code{aperm} for Tensor class for convenience.
}
\details{
  \code{tperm(x,perm=NULL,...)}
}
\examples{
tnsr <- rand_tensor(c(3,4,5))
dim(tperm(tnsr,perm=c(2,1,3)))
dim(tperm(tnsr,perm=c(1,3,2)))
}

