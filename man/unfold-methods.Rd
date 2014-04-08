\docType{methods}
\name{unfold-methods}
\alias{unfold}
\alias{unfold,Tensor-method}
\alias{unfold-methods}
\alias{vec}
\alias{vec,Tensor-method}
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
  elements (order doesn't matter) as \code{x@modes}. Within
  the rows and columns, the order of the unfolding is
  determined by the order of the modes. This convention is
  consistent with Kolda and Bader (2009).
}
\details{
  For Row Space Unfolding or m-mode Unfolding, see
  \code{\link{rs_unfold-methods}}. For Column Space
  Unfolding or matvec, see \code{\link{cs_unfold-methods}}.

  \code{vec} returns the vectorization of the tensor.

  \code{unfold(tnsr,rs=NULL,cs=NULL)}
}
\examples{
tnsr <- rand_tensor()
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

