\name{cs_fold}
\alias{cs_fold}
\title{Column Space Folding of Matrix}
\usage{
  cs_fold(mat, m = NULL, modes = NULL)
}
\arguments{
  \item{mat}{matrix to be folded into a Tensor}

  \item{m}{the index of the mode that is mapped onto the
  column space}

  \item{modes}{the modes of the output Tensor}
}
\value{
  Tensor object with modes given by \code{modes}
}
\description{
  Column space folding of a matrix into a Tensor. This is
  the inverse funtion to \code{cs_unfold} in the m mode. In
  particular, \code{cs_fold(cs_unfold(tnsr,
  m),m,getModes(tnsr))} will result in the original Tensor.
  For a full account of matrix folding/unfolding
  operations, consult Kolda and Bader (2009).
}
\details{
  This is a wrapper function to \code{\link{fold}}.
}
\examples{
tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
matT1<-cs_unfold(tnsr,m=3)
identical(cs_fold(matT1,m=3,modes=c(3,4,5)),tnsr)
}
\references{
  T. Kolda, B. Bader, "Tensor decomposition and
  applications". SIAM Applied Mathematics and Applications
  2009.
}
\seealso{
  \code{\link{cs_unfold-methods}}, \code{\link{fold}},
  \code{\link{rs_fold}}
}

