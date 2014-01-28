\docType{methods}
\name{sweep-methods}
\alias{sweep,Tensor-method}
\alias{sweep-methods}
\title{Sweep for Tensor}
\arguments{
  \item{x}{the Tensor instance}

  \item{...}{additional parameters to be passed into
  sweep()}
}
\description{
  Extend sweep for Tensor
}
\details{
  \code{sweep(x,m=NULL,stats=NULL,func=NULL,...)}
}
\examples{
tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
sweep(tnsr,m=c(2,3),stat=1,func='-')
sweep(tnsr,m=1,stat=10,func='/')
}
\seealso{
  \code{\link{modeSum}} and \code{\link{modeMean}}
}

