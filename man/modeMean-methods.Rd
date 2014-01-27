\docType{methods}
\name{modeMean-methods}
\alias{modeMean}
\alias{modeMean,Tensor-method}
\alias{modeMean-methods}
\title{Tensor Mean Across Single Mode}
\arguments{
  \item{tnsr}{the Tensor instance}

  \item{m}{the index of the mode to average across}
}
\value{
  K-1 Tensor, where \code{K = getNumModes(x)}
}
\description{
  Given a mode for a K-tensor, this returns the K-1 tensor
  resulting from taking the mean across that particular
  mode.
}
\details{
  \code{modeMean(tnsr,m=NULL)}
}
\examples{
tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
modeMean(tnsr,1)
}
\seealso{
  \code{\link{modeSum}} and \code{\link{sweep}}
}

