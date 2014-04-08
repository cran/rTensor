\docType{methods}
\name{fnorm-methods}
\alias{fnorm}
\alias{fnorm,Tensor-method}
\alias{fnorm-methods}
\title{Tensor Frobenius Norm}
\arguments{
  \item{tnsr}{the Tensor instance}
}
\value{
  numeric Frobenius norm of \code{x}
}
\description{
  Returns the Frobenius norm of the Tensor instance.
}
\details{
  \code{fnorm(tnsr)}
}
\examples{
tnsr <- rand_tensor()
fnorm(tnsr)
}

