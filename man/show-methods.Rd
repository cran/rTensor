\docType{methods}
\name{show-methods}
\alias{show,Tensor-method}
\alias{show-methods}
\title{Show for Tensor}
\arguments{
  \item{x}{the Tensor instance}

  \item{...}{additional parameters to be passed into
  show()}
}
\description{
  Extend show for Tensor
}
\details{
  \code{show(x)}
}
\examples{
tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
tnsr
}
\seealso{
  \code{\link{print}}
}

