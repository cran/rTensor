\docType{methods}
\name{rs_unfold-methods}
\alias{rs_unfold}
\alias{rs_unfold,Tensor-method}
\alias{rs_unfold-methods}
\title{Tensor Row Space Unfolding}
\arguments{
  \item{x}{the Tensor instance}

  \item{m}{the index of the mode to map onto the row space}
}
\value{
  atrix with \code{x@modes[m]} rows and
  \code{prod(x@modes[-m])} columns
}
\description{
  Unfolding of a tensor by mapping the mode 'm' onto the
  row space, and all other modes onto the column space.
  This the most common type of unfolding operation for
  Tucker decompositions and its variants. Also known as
  m-Mode unfolding/Matricization.
}
\details{
  \code{rs_unfold(tnsr,m=NULL)}
}
\examples{
tnsr <- rand_tensor()
matT2<-rs_unfold(tnsr,m=2)
}
\references{
  T. Kolda and B. Bader, "Tensor decomposition and
  applications". SIAM Applied Mathematics and Applications
  2009.
}
\seealso{
  \code{\link{cs_unfold}} and \code{\link{unfold}}
}

