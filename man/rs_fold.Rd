\name{rs_fold}
\alias{rs_fold}
\title{Row Space Folding of Matrix}
\usage{
  rs_fold(mat, m = NULL, modes = NULL)
}
\arguments{
  \item{mat}{matrix to be folded into a Tensor}

  \item{m}{the index of the mode that is mapped onto the
  row space}

  \item{modes}{the modes of the output Tensor}
}
\value{
  Tensor object with modes given by \code{modes}
}
\description{
  Row space folding of a matrix into a Tensor. This is the
  inverse funtion to \code{rs_unfold} in the m mode. In
  particular, \code{rs_fold(rs_unfold(tnsr,
  m),m,getModes(tnsr))} will result in the original Tensor.
  For a full account of matrix folding/unfolding
  operations, consult Kolda and Bader (2009).
}
\details{
  This is a wrapper function to \code{\link{fold}}.
}
\examples{
tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
matT2<-rs_unfold(tnsr,m=2)
identical(rs_fold(matT2,m=2,modes=c(3,4,5)),tnsr)
}
\references{
  T. Kolda, B. Bader, "Tensor decomposition and
  applications". SIAM Applied Mathematics and Applications
  2009.
}
\seealso{
  \code{\link{rs_unfold-methods}}, \code{\link{fold}},
  \code{\link{cs_fold}}
}

