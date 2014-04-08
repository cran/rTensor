\docType{methods}
\name{cs_unfold-methods}
\alias{cs_unfold}
\alias{cs_unfold,Tensor-method}
\alias{cs_unfold-methods}
\alias{matvec}
\alias{matvec,Tensor-method}
\title{Tensor Column Space Unfolding}
\arguments{
  \item{tnsr}{the Tensor instance}

  \item{m}{the index of the mode to map onto the column
  space}
}
\value{
  matrix with \code{prod(x@modes[-m])} rows and
  \code{x@modes[m]} columns
}
\description{
  Unfolding of a tensor by mapping the mode 'm' onto the
  column space, and all other modes onto the row space. For
  3-tensors, this is also known as the 'matvec' operation
  when \code{m=2}. In fact, we provide \code{matvec} for
  convenience as well. This is the prevalent unfolding for
  T-SVD and T-MULT based on block circulant matrices.
}
\details{
  \code{cs_unfold(tnsr,m=NULL)}
}
\examples{
tnsr <- rand_tensor()
matT1<-cs_unfold(tnsr,m=3)
}
\references{
  M. Kilmer, K. Braman, N. Hao, and R. Hoover, "Third-order
  tensors as operators on matrices: a theoretical and
  computational framework with applications in imaging".
  SIAM Journal on Matrix Analysis and Applications 2013.
}
\seealso{
  \code{\link{rs_unfold-methods}} and
  \code{\link{unfold-methods}}
}

