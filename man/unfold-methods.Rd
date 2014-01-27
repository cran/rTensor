\docType{methods}
\name{unfold-methods}
\alias{unfold}
\alias{unfold,Tensor-method}
\alias{unfold-methods}
\title{Tensor Unfolding}
\arguments{
  \item{tnsr}{the Tensor instance}

  \item{rs}{the indices of the modes to map onto the row
  space}

  \item{cs}{the indices of the modes to map onto the column
  space}
}
\value{
  matrix with \code{prod(rs)} rows and \code{prod(cs)}
  columns
}
\description{
  Unfolds the tensor into a matrix, with the modes in
  \code{rs} onto the rows and modes in \code{cs} onto the
  columns. Note that \code{c(rs,cs)} must have the same
  elements (order doesn't matter) as \code{getModes(x)}.
  Within the rows and columns, the order of the unfolding
  is determined by the order of the modes. This convention
  is consistent with Kolda and Bader (2009).
}
\details{
  \code{unfold(tnsr,rs=NULL,cs=NULL)}
}
\examples{
tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
matT3<-unfold(tnsr,rs=2,cs=c(3,1))
}
\references{
  T. Kolda, B. Bader, "Tensor decomposition and
  applications". SIAM Applied Mathematics and Applications
  2009.
}
\seealso{
  \code{\link{cs_unfold-methods}} and
  \code{\link{rs_unfold-methods}}
}

