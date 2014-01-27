\docType{methods}
\name{modeSum-methods}
\alias{modeSum}
\alias{modeSum,Tensor-method}
\alias{modeSum-methods}
\title{Tensor Sum Across Single Mode}
\arguments{
  \item{tnsr}{the Tensor instance}

  \item{m}{the index of the mode to sum across}
}
\value{
  K-1 tensor, where \code{K = getNumModes(x)}
}
\description{
  Given a mode for a K-tensor, this returns the K-1 tensor
  resulting from summing across that particular mode.
}
\details{
  \code{modeSum(tnsr,m=NULL)}
}
\examples{
tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
modeSum(tnsr,3)
}
\seealso{
  \code{\link{modeMean}} and \code{\link{sweep}}
}

