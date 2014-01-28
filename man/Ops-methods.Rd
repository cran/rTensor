\docType{methods}
\name{Ops-methods}
\alias{Ops,Tensor,Tensor-method}
\alias{Ops,Tensor,array-method}
\alias{Ops,Tensor,numeric-method}
\alias{Ops,array,Tensor-method}
\alias{Ops,numeric,Tensor-method}
\alias{Ops-methods}
\title{Conformable elementwise operators for Tensor}
\description{
  Overloads elementwise operators for tensors, arrays, and
  vectors that are conformable (have the same modes).
}
\examples{
tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
tnsr2 <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
tnsrsum <- tnsr + tnsr2
tnsrdiff <- tnsr - tnsr2
tnsrelemprod <- tnsr * tnsr2
tnsrelemquot <- tnsr / tnsr2
for (i in 1:3L){
	for (j in 1:4L){
		for (k in 1:5L){
			stopifnot(tnsrsum@data[i,j,k]==tnsr@data[i,j,k]+tnsr2@data[i,j,k])
			stopifnot(tnsrdiff@data[i,j,k]==(tnsr@data[i,j,k]-tnsr2@data[i,j,k]))
			stopifnot(tnsrelemprod@data[i,j,k]==tnsr@data[i,j,k]*tnsr2@data[i,j,k])
			stopifnot(tnsrelemquot@data[i,j,k]==tnsr@data[i,j,k]/tnsr2@data[i,j,k])
}
}
}
}

