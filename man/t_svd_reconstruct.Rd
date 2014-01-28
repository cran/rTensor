\name{t_svd_reconstruct}
\alias{t_svd_reconstruct}
\title{Reconstruct Tensor From TSVD}
\usage{
  t_svd_reconstruct(L)
}
\arguments{
  \item{L}{list that is an output from \code{\link{t_svd}}}
}
\value{
  a 3-Tensor
}
\description{
  Reconstruct the original 3-Tensor after it has been
  decomposed into \code{U, S, V} via \code{\link{t_svd}}.
}
\examples{
tnsr <- new("Tensor",3L,c(10L,10L,100L),data=runif(10000))
tsvdD <- t_svd(tnsr)
1 - fnorm(t_svd_reconstruct(tsvdD)-tnsr)/fnorm(tnsr)
smalltnsr <- new("Tensor",3L,c(10L,10L,10L),data=runif(1000))
smalltsvdD <- t_svd(smalltnsr)
1 - fnorm(t_svd_reconstruct(smalltsvdD)-smalltnsr)/fnorm(smalltnsr)
}
\seealso{
  \code{\link{t_svd}}
}

