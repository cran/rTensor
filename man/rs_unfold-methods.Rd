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
  atrix with \code{getModes(x)[m]} rows and
  \code{prod(getModes(x)[-m])} columns
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
tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
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

