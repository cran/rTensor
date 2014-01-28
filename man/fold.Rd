\name{fold}
\alias{fold}
\title{General Folding of Matrix}
\usage{
  fold(mat, rs = NULL, cs = NULL, modes = NULL)
}
\arguments{
  \item{mat}{matrix to be folded into a Tensor}

  \item{rs}{the indices of the modes that are mapped onto
  the row space}

  \item{cs}{the indices of the modes that are mapped onto
  the column space}

  \item{modes}{the modes of the output Tensor}
}
\value{
  Tensor object with modes given by \code{modes}
}
\description{
  General folding of a matrix into a Tensor. This is
  designed to be the inverse function to
  \code{\link{unfold-methods}}, with the same ordering of
  the indices. This amounts to following: if we were to
  unfold a Tensor using a set of \code{rs} and \code{cs},
  then we can fold the resulting matrix back into the
  original Tensor using the same \code{rs} and \code{cs}.
  See Kolda and Bader (2009) for more details.
}
\details{
  This function uses \code{aperm} as the primary workhorse.
}
\examples{
tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
matT3<-unfold(tnsr,rs=2,cs=c(3,1))
identical(fold(matT3,rs=2,cs=c(3,1),modes=c(3,4,5)),tnsr)
}
\references{
  T. Kolda, B. Bader, "Tensor decomposition and
  applications". SIAM Applied Mathematics and Applications
  2009.
}
\seealso{
  \code{\link{unfold-methods}}, \code{\link{rs_fold}},
  \code{\link{cs_fold}}
}

