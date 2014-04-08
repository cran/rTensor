###Class Definition

#'S4 Class for a Tensor
#'
#'An S4 class for a tensor with arbitrary number of modes. The Tensor class extends the base 'array' class to include additional tensor manipulation (folding, unfolding, reshaping, subsetting) as well as a formal class definition that enables more explicit tensor algebra.
#'
#'@section Slots:
#' \describe{
#'	\item{num_modes}{number of modes (integer)}
#'  \item{modes}{vector of modes (integer), aka sizes/extents/dimensions}
#'  \item{data}{actual data of the tensor, which can be 'array' or 'vector'}
#' }
#'@name Tensor-class
#'@rdname Tensor-class
#'@aliases Tensor Tensor-class 
#'@docType class
#'@exportClass Tensor
#'@section Methods:
#'  \describe{
#'    \item{[}{\code{signature(tnsr = "Tensor")}: ... }
#'    \item{[<-}{\code{signature(tnsr = "Tensor")}: ... }
#'    \item{cs_unfold}{\code{signature(tnsr = "Tensor")}: ... }
#'    \item{dim}{\code{signature(tnsr = "Tensor")}: ... }
#'    \item{fnorm}{\code{signature(tnsr = "Tensor")}: ... }
#'    \item{head}{\code{signature(tnsr = "Tensor")}: ... }
#'    \item{initialize}{\code{signature(.Object = "Tensor")}: ... }
#'    \item{innerProd}{\code{signature(tnsr1 = "Tensor", tnsr2 = "Tensor")}: ... }
#'    \item{modeMean}{\code{signature(tnsr = "Tensor")}: ... }
#'    \item{modeSum}{\code{signature(tnsr = "Tensor")}: ... }
#'    \item{Ops}{\code{signature(e1 = "array", e2 = "Tensor")}: ... }
#'    \item{Ops}{\code{signature(e1 = "numeric", e2 = "Tensor")}: ... }
#'    \item{Ops}{\code{signature(e1 = "Tensor", e2 = "array")}: ... }
#'    \item{Ops}{\code{signature(e1 = "Tensor", e2 = "numeric")}: ... }
#'    \item{Ops}{\code{signature(e1 = "Tensor", e2 = "Tensor")}: ... }
#'    \item{print}{\code{signature(tnsr = "Tensor")}: ... }
#'    \item{rs_unfold}{\code{signature(tnsr = "Tensor")}: ... }
#'    \item{show}{\code{signature(tnsr = "Tensor")}: ... }
#'    \item{t}{\code{signature(tnsr = "Tensor")}: ... }
#'    \item{tail}{\code{signature(tnsr = "Tensor")}: ... }
#'    \item{unfold}{\code{signature(tnsr = "Tensor")}: ... }
#'    \item{tperm}{\code{signature(tnsr = "Tensor")}: ...}
#'    \item{image}{\code{signature(tnsr = "Tensor")}: ...}
#'	 }
#'@author James Li \email{jamesyili@@gmail.com}
#'@details {This can be seen as a wrapper class to the base \code{array} class. While it is possible to create an instance using \code{new}, it is also possible to do so by passing the data into \code{\link{as.tensor}}.
#'	
#'Each slot of a Tensor instance can be obtained using \code{@@}.
#'
#'The following methods are overloaded for the Tensor class: \code{\link{dim-methods}}, \code{\link{head-methods}}, \code{\link{tail-methods}}, \code{\link{print-methods}}, \code{\link{show-methods}},  element-wise array operations, array subsetting (extract via `['), array subset replacing (replace via `[<-'), and \code{\link{tperm-methods}}, which is a wrapper around the base \code{aperm} method.
#'
#'To sum across any one mode of a tenor, use the function \code{\link{modeSum-methods}}. To compute the mean across any one mode, use \code{\link{modeMean-methods}}.
#'
#'You can always unfold any Tensor into a matrix, and the \code{\link{unfold-methods}}, \code{\link{rs_unfold-methods}}, and \code{\link{cs_unfold-methods}} methods are for that purpose. The output can be kept as a Tensor with 2 modes or a \code{matrix} object. Note that the m-mode unfolding is the same as rs_unfold for mode m, and matvec is the same as cs_unfold for mode 2. The vectorization function is also provided as \code{vec}.
#'
#'Conversion from \code{array}/\code{matrix} to Tensor is facilitated via \code{\link{as.tensor}}. To convert from a Tensor instance, simply invoke \code{@@data}.
#'
#'The Frobenius norm of the Tensor is given by \code{\link{fnorm-methods}}, while the inner product between two Tensors (of equal modes) is given by \code{\link{innerProd-methods}}. You can also sum through any one mode to obtain the K-1 Tensor sum using \code{\link{modeSum-methods}}. \code{\link{modeMean-methods}} provides similar functionality to obtain the K-1 Tensor mean. These are primarily meant to be used internally but may be useful in doing statistics with Tensors.
#'
#'For Tensors with 3 modes, we also overloaded \code{t} (transpose) and \code{\link{image-methods}}, defined by Kilmer et.al (2013). See \code{\link{t-methods}}.
#'
#'To create a Tensor with i.i.d. random normal(0, 1) entries, see \code{\link{rand_tensor}}.
#'}
#'@note All of the decompositions and regression models in this package require a Tensor input.
#'@references M. Kilmer, K. Braman, N. Hao, and R. Hoover, "Third-order tensors as operators on matrices: a theoretical and computational framework with applications in imaging". SIAM Journal on Matrix Analysis and Applications 2013.
#'@seealso \code{\link{as.tensor}}
#'@examples
#'tnsr <- rand_tensor()
#'class(tnsr)
#'tnsr
#'print(tnsr)
#'dim(tnsr)
#'tnsr@@num_modes
#'tnsr@@data
setClass("Tensor",
representation(num_modes = "integer", modes = "integer", data="array"),
validity = function(object){
	num_modes <- object@num_modes
	modes <- object@modes
	errors <- character()
	if (any(modes <= 0)){
		msg <- "'modes' must contain strictly positive values; if any mode is 1, consider a smaller num_modes"
		errors <- c(errors, msg)
	}
	if(length(errors)==0) TRUE else errors
})

###Generic Definitions

#'Tensor Unfolding
#'
#'Unfolds the tensor into a matrix, with the modes in \code{rs} onto the rows and modes in \code{cs} onto the columns. Note that \code{c(rs,cs)} must have the same elements (order doesn't matter) as \code{x@@modes}. Within the rows and columns, the order of the unfolding is determined by the order of the modes. This convention is consistent with Kolda and Bader (2009). 
#'
#'For Row Space Unfolding or m-mode Unfolding, see \code{\link{rs_unfold-methods}}. For Column Space Unfolding or matvec, see \code{\link{cs_unfold-methods}}.
#'
#'\code{vec} returns the vectorization of the tensor.
#'
#'@details \code{unfold(tnsr,rs=NULL,cs=NULL)}
#'@export
#'@docType methods
#'@name unfold-methods
#'@rdname unfold-methods
#'@aliases unfold unfold,Tensor-method
#'@references T. Kolda, B. Bader, "Tensor decomposition and applications". SIAM Applied Mathematics and Applications 2009.
#'@param tnsr the Tensor instance
#'@param rs the indices of the modes to map onto the row space
#'@param cs the indices of the modes to map onto the column space
#'@return matrix with \code{prod(rs)} rows and \code{prod(cs)} columns
#'@seealso \code{\link{cs_unfold-methods}} and \code{\link{rs_unfold-methods}}
#'@examples
#'tnsr <- rand_tensor()
#'matT3<-unfold(tnsr,rs=2,cs=c(3,1))
setGeneric(name="unfold",
def=function(tnsr,rs,cs){standardGeneric("unfold")})

#'Tensor Row Space Unfolding
#'
#'Unfolding of a tensor by mapping the mode 'm' onto the row space, and all other modes onto the column space. This the most common type of unfolding operation for Tucker decompositions and its variants. Also known as m-Mode unfolding/Matricization. 
#'
#'@docType methods
#'@name rs_unfold-methods
#'@details \code{rs_unfold(tnsr,m=NULL)}
#'@export
#'@rdname rs_unfold-methods
#'@aliases rs_unfold rs_unfold,Tensor-method
####aliases rs_unfold,ANY-method
#'@references T. Kolda and B. Bader, "Tensor decomposition and applications". SIAM Applied Mathematics and Applications 2009.
#'@param x the Tensor instance
#'@param m the index of the mode to map onto the row space
#'@return atrix with \code{x@@modes[m]} rows and \code{prod(x@@modes[-m])} columns
#'@seealso \code{\link{cs_unfold}} and \code{\link{unfold}}
#'@examples
#'tnsr <- rand_tensor()
#'matT2<-rs_unfold(tnsr,m=2)
setGeneric(name="rs_unfold",
def=function(tnsr,m){standardGeneric("rs_unfold")})

#'Tensor Column Space Unfolding
#'
#'Unfolding of a tensor by mapping the mode 'm' onto the column space, and all other modes onto the row space. For 3-tensors, this is also known as the 'matvec' operation when \code{m=2}. In fact, we provide \code{matvec} for convenience as well. This is the prevalent unfolding for T-SVD and T-MULT based on block circulant matrices.
#'@docType methods
#'@name cs_unfold-methods
#'@details \code{cs_unfold(tnsr,m=NULL)}
#'@export
#'@rdname cs_unfold-methods
#'@aliases cs_unfold cs_unfold,Tensor-method
#'@references M. Kilmer, K. Braman, N. Hao, and R. Hoover, "Third-order tensors as operators on matrices: a theoretical and computational framework with applications in imaging". SIAM Journal on Matrix Analysis and Applications 2013.
#'@param tnsr the Tensor instance
#'@param m the index of the mode to map onto the column space
#'@return matrix with \code{prod(x@@modes[-m])} rows and \code{x@@modes[m]} columns
#'@seealso \code{\link{rs_unfold-methods}} and \code{\link{unfold-methods}}
#'@examples
#'tnsr <- rand_tensor()
#'matT1<-cs_unfold(tnsr,m=3)
setGeneric(name="cs_unfold",
def=function(tnsr,m){standardGeneric("cs_unfold")})

#'Tensor Sum Across Single Mode
#'
#'Given a mode for a K-tensor, this returns the K-1 tensor resulting from summing across that particular mode.
#'
#'@docType methods
#'@name modeSum-methods
#'@details \code{modeSum(tnsr,m=NULL,drop=FALSE)}
#'@export
#'@rdname modeSum-methods
#'@aliases modeSum modeSum,Tensor-method
#'@param tnsr the Tensor instance
#'@param m the index of the mode to sum across
#'@param drop whether or not mode m should be dropped
#'@return K-1 or K tensor, where \code{K = x@@num_modes}
#'@seealso \code{\link{modeMean}}
#'@examples
#'tnsr <- rand_tensor()
#'modeSum(tnsr,3,drop=TRUE)
setGeneric(name="modeSum",
def=function(tnsr,m,drop){standardGeneric("modeSum")})

#'Tensor Mean Across Single Mode
#'
#'Given a mode for a K-tensor, this returns the K-1 tensor resulting from taking the mean across that particular mode.
#'
#'@docType methods
#'@name modeMean-methods
#'@details \code{modeMean(tnsr,m=NULL,drop=FALSE)}
#'@export
#'@rdname modeMean-methods
#'@aliases modeMean modeMean,Tensor-method
#'@param tnsr the Tensor instance
#'@param m the index of the mode to average across
#'@param drop whether or not mode m should be dropped
#'@return K-1 or K Tensor, where \code{K = x@@num_modes}
#'@seealso \code{\link{modeSum}}
#'@examples
#'tnsr <- rand_tensor()
#'modeMean(tnsr,1,drop=TRUE)
setGeneric(name="modeMean",
def=function(tnsr,m,drop){standardGeneric("modeMean")})

#'Tensor Frobenius Norm
#'
#'Returns the Frobenius norm of the Tensor instance.
#'
#'@docType methods
#'@name fnorm-methods
#'@details \code{fnorm(tnsr)}
#'@export
#'@rdname fnorm-methods
#'@aliases fnorm fnorm,Tensor-method
#'@param tnsr the Tensor instance
#'@return numeric Frobenius norm of \code{x}
#'@examples
#'tnsr <- rand_tensor()
#'fnorm(tnsr)
setGeneric(name="fnorm",
def=function(tnsr){standardGeneric("fnorm")})

#'Tensors Inner Product
#'
#'Returns the inner product between two Tensors
#'
#'@docType methods
#'@name innerProd-methods
#'@details \code{innerProd(tnsr1,tnsr2)}
#'@export
#'@rdname innerProd-methods
#'@aliases innerProd innerProd,Tensor,Tensor-method
#'@param tnsr1 first Tensor instance
#'@param tnsr2 second Tensor instance
#'@return inner product between \code{x1} and \code{x2}
#'@examples
#'tnsr1 <- rand_tensor()
#'tnsr2 <- rand_tensor()
#'innerProd(tnsr1,tnsr2)
setGeneric(name="innerProd",
def=function(tnsr1,tnsr2){standardGeneric("innerProd")})

#'Initializes a Tensor instance
#'
#'Not designed to be called by the user. Use \code{as.tensor} instead.
#' 
#'@docType methods
#'@name initialize-methods
#'@rdname initialize-methods
#'@aliases initialize,Tensor-method
#'@seealso \code{as.tensor}
setMethod(f="initialize",
signature="Tensor",
definition = function(.Object, num_modes=NULL, modes=NULL, data=NULL){
	if(is.null(num_modes)){
		if (is.vector(data)) num_modes <- 1L
		else{num_modes <- length(dim(data))}
	}
	if(is.null(modes)){
		if (is.vector(data)) modes <- length(data)
		else{modes <- dim(data)}
	}
	.Object@num_modes <- num_modes
	.Object@modes <- modes
	.Object@data <- array(data,dim=modes)
	validObject(.Object)
	.Object
})

###Method Definitions
options(warn=-1)

#'Mode Getter for Tensor
#'
#'Return the vector of modes from a tensor
#'
#'@name dim-methods
#'@details \code{dim(tnsr)}
#'@export
#'@aliases dim,Tensor-method
#'@docType methods
#'@rdname dim-methods
#'@param tnsr the Tensor instance
#'@return an integer vector of the modes associated with \code{x}
#'@examples
#'tnsr <- rand_tensor()
#'dim(tnsr)
setMethod(f="dim",
signature="Tensor",
definition=function(tnsr){
	tnsr@modes
})

#'Show for Tensor
#'
#'Extend show for Tensor
#'
#'@name show-methods
#'@details \code{show(x)}
#'@export
#'@aliases show,Tensor-method
#'@docType methods
#'@rdname show-methods
#'@param x the Tensor instance
#'@param ... additional parameters to be passed into show()
#'@seealso \code{\link{print}}
#'@examples
#'tnsr <- rand_tensor()
#'tnsr
setMethod(f="show",
signature="Tensor",
definition=function(x){
	cat("Numeric Tensor of", x@num_modes, "Modes\n", sep=" ")
	cat("Modes: ", x@modes, "\n", sep=" ")
	cat("Data: \n")
	print(head(x@data))
})

#'Print for Tensor
#'
#'Extend print for Tensor
#'
#'@name print-methods
#'@details \code{print(x,...)}
#'@export
#'@aliases print,Tensor-method
#'@docType methods
#'@rdname print-methods
#'@param x the Tensor instance
#'@param ... additional parameters to be passed into print()
#'@seealso \code{\link{show}}
#'@examples
#'tnsr <- rand_tensor()
#'print(tnsr)
setMethod(f="print",
signature="Tensor",
definition=function(x,...){
	show(x)
})

#'Head for Tensor
#'
#'Extend head for Tensor
#'
#'@name head-methods
#'@details \code{head(x,...)}
#'@export
#'@aliases head,Tensor-method
#'@docType methods
#'@rdname head-methods
#'@param x the Tensor instance
#'@param ... additional parameters to be passed into head()
#'@seealso \code{\link{tail-methods}}
#'@examples
#'tnsr <- rand_tensor()
#'head(tnsr)
setMethod(f="head",
signature="Tensor",
definition=function(x,...){
	head(x@data,...)
})

#'Tail for Tensor
#'
#'Extend tail for Tensor
#'
#'@name tail-methods
#'@details \code{tail(x,...)}
#'@export
#'@aliases tail,Tensor-method
#'@docType methods
#'@rdname tail-methods
#'@param x the Tensor instance
#'@param ... additional parameters to be passed into tail()
#'@seealso \code{\link{head-methods}}
#'@examples
#'tnsr <- rand_tensor()
#'tail(tnsr)
setMethod(f="tail",
signature="Tensor",
definition=function(x,...){
	tail(x@data,...)
})

#'Extract or Replace Subtensors
#'
#'Extends '[' and '[<-' from the base array class for the Tensor class. Works exactly as it would for the base 'array' class.
#'
#'@name [-methods
#'@details \code{x[i,j,...,drop=TRUE]}
#'@export
#'@aliases [,Tensor-method extract,Tensor-method [<-,Tensor-method
#'@docType methods
#'@rdname extract-methods
#'@param x Tensor to be subset
#'@param i,j,... indices that specify the extents of the sub-tensor
#'@param drop whether or not to reduce the number of modes to exclude those that have '1' as the mode
#'@param value either vector, matrix, or array that will replace the subtensor
#'@return an object of class Tensor
#'@examples
#'tnsr <- rand_tensor()
#'tnsr[1,2,3]
#'tnsr[3,1,]
#'tnsr[,,5]
#'tnsr[,,5,drop=FALSE]
#'
#'tnsr[1,2,3] <- 3; tnsr[1,2,3]
#'tnsr[3,1,] <- rep(0,5); tnsr[3,1,]
#'tnsr[,2,] <- matrix(0,nrow=3,ncol=5); tnsr[,2,]
setMethod("[", signature="Tensor",
definition=function(x,i,j,...,drop=TRUE){
	if(!drop) as.tensor(`[`(x@data,i,j,drop=FALSE,...),drop=drop)
	else as.tensor(`[`(x@data,i,j,...))
})

#'@aliases [,Tensor-method extract,Tensor-method [<-,Tensor-method
#'@rdname extract-methods
setMethod("[<-", signature="Tensor",
definition=function(x,i,j,...,value){
	as.tensor(`[<-`(x@data,i,j,...,value=value))
})

#'Tensor Transpose
#'
#'Implements the tensor transpose based on block circulant matrices (Kilmer et al. 2013) for 3-tensors.
#'
#'@docType methods
#'@name t-methods
#'@rdname t-methods
#'@details \code{t(tnsr)}
#'@export
#'@aliases t,Tensor-method
#'@param tnsr a 3-tensor
#'@return tensor transpose of \code{x}
#'@references M. Kilmer, K. Braman, N. Hao, and R. Hoover, "Third-order tensors as operators on matrices: a theoretical and computational framework with applications in imaging". SIAM Journal on Matrix Analysis and Applications 2013.
#'@examples
#'tnsr <- rand_tensor()
#'identical(t(tnsr)@@data[,,1],t(tnsr@@data[,,1]))
#'identical(t(tnsr)@@data[,,2],t(tnsr@@data[,,5]))
#'identical(t(t(tnsr)),tnsr)
setMethod("t",signature="Tensor",
definition=function(tnsr){
	if(tnsr@num_modes!=3) stop("Tensor Transpose currently only implemented for 3d Tensors")
	modes <- tnsr@modes
	new_arr <- array(apply(tnsr@data[,,c(1L,modes[3]:2L),drop=FALSE],MARGIN=3,FUN=t),dim=modes[c(2,1,3)])
	as.tensor(new_arr)
})

#'Conformable elementwise operators for Tensor
#'
#'Overloads elementwise operators for tensors, arrays, and vectors that are conformable (have the same modes).
#'
#'@export
#'@name Ops-methods
#'@docType methods
#'@aliases Ops-methods
#'@aliases Ops,Tensor,Tensor-method
#'@aliases Ops,Tensor,array-method
#'@aliases Ops,Tensor,numeric-method
#'@aliases Ops,array,Tensor-method
#'@aliases Ops,numeric,Tensor-method
#'@examples
#'tnsr <- rand_tensor()
#'tnsr2 <- rand_tensor()
#'tnsrsum <- tnsr + tnsr2
#'tnsrdiff <- tnsr - tnsr2
#'tnsrelemprod <- tnsr * tnsr2
#'tnsrelemquot <- tnsr / tnsr2
#'for (i in 1:3L){
#'	for (j in 1:4L){
#'		for (k in 1:5L){
#'			stopifnot(tnsrsum@@data[i,j,k]==tnsr@@data[i,j,k]+tnsr2@@data[i,j,k])
#'			stopifnot(tnsrdiff@@data[i,j,k]==(tnsr@@data[i,j,k]-tnsr2@@data[i,j,k]))
#'			stopifnot(tnsrelemprod@@data[i,j,k]==tnsr@@data[i,j,k]*tnsr2@@data[i,j,k])
#'			stopifnot(tnsrelemquot@@data[i,j,k]==tnsr@@data[i,j,k]/tnsr2@@data[i,j,k])
#'}
#'}
#'}
setMethod("Ops", signature(e1="Tensor", e2="Tensor"),
definition=function(e1,e2){
	e1@data<-callGeneric(e1@data, e2@data)
	validObject(e1)
	e1
})
setMethod("Ops", signature(e1="Tensor", e2="array"),
definition=function(e1,e2){
	e1@data<-callGeneric(e1@data,e2)
	validObject(e1)
	e1
})
setMethod("Ops", signature(e1="array", e2="Tensor"),
definition=function(e1,e2){
	e2@data<-callGeneric(e1,e2@data)
	validObject(e2)
	e2
})
setMethod("Ops", signature(e1="Tensor", e2="numeric"),
definition=function(e1,e2){
	e1@data<-callGeneric(e1@data,e2)
	validObject(e1)
	e1
})
setMethod("Ops", signature(e1="numeric", e2="Tensor"),
definition=function(e1,e2){
	e2@data<-callGeneric(e1,e2@data)
	validObject(e2)
	e2
})

#'@rdname modeSum-methods
#'@aliases modeSum,Tensor-method
setMethod("modeSum",signature="Tensor",
definition=function(tnsr,m=NULL,drop=FALSE){
	if(is.null(m)) stop("must specify mode m")
	num_modes <- tnsr@num_modes
	if(m<1||m>num_modes) stop("m out of bounds")
	perm <- c(m,(1L:num_modes)[-m])
	modes <- tnsr@modes
	newmodes <- modes; newmodes[m]<-1
	arr <- array(colSums(aperm(tnsr@data,perm),dims=1L),dim=newmodes)
	as.tensor(arr,drop=drop)
})

#'@rdname modeMean-methods
#'@aliases modeMean,Tensor-method
setMethod("modeMean",signature="Tensor",
definition=function(tnsr,m=NULL,drop=FALSE){
	if(is.null(m)) stop("must specify mode m")
	num_modes <- tnsr@num_modes
	if(m<1||m>num_modes) stop("m out of bounds")
	perm <- c(m,(1L:num_modes)[-m])
	modes <- tnsr@modes
	newmodes <- modes; newmodes[m]<-1
	arr <- array(colSums(aperm(tnsr@data,perm),dims=1L),dim=newmodes)
	as.tensor(arr/modes[m],drop=drop)
})

#'@rdname fnorm-methods
#'@aliases fnorm,Tensor-method
setMethod("fnorm",signature="Tensor",
definition=function(tnsr){
	arr<-tnsr@data
	sqrt(sum(arr*arr))
})

#'@rdname innerProd-methods
#'@aliases innerProd,Tensor,Tensor-method
setMethod("innerProd",signature=c(tnsr1="Tensor", tnsr2="Tensor"),
definition=function(tnsr1,tnsr2){
	stopifnot(tnsr1@modes==tnsr2@modes)
	arr1 <- tnsr1@data
	arr2 <- tnsr2@data
	sum(as.numeric(arr1*arr2))
})

###Tensor Unfoldings

#'@rdname unfold-methods
#'@aliases unfold,Tensor-method
setMethod("unfold", signature="Tensor",
definition=function(tnsr,rs=NULL,cs=NULL){
	#checks
	if(is.null(rs)||is.null(cs)) stop("row space and col space indices must be specified")
	num_modes <- tnsr@num_modes
	if (length(rs) + length(cs) != num_modes) stop("incorrect number of indices")
	if(any(rs<1) || any(rs>num_modes) || any(cs < 1) || any(cs>num_modes)) stop("illegal indices specified")
	perm <- c(rs,cs)
	if (any(sort(perm,decreasing=TRUE) != num_modes:1)) stop("missing and/or repeated indices")
	modes <- tnsr@modes
	mat <- tnsr@data
	new_modes <- c(prod(modes[rs]),prod(modes[cs]))
	#rearranges into a matrix
	mat <- aperm(mat,perm)
	dim(mat) <- new_modes
	as.tensor(mat)
})

#'@rdname rs_unfold-methods
#'@aliases rs_unfold,Tensor-method
setMethod("rs_unfold", signature="Tensor",
definition=function(tnsr,m=NULL){
	if(is.null(m)) stop("mode m must be specified")
	num_modes <- tnsr@num_modes
	rs <- m
	cs <- (1:num_modes)[-m]
	unfold(tnsr,rs=rs,cs=cs)
})

#'@rdname cs_unfold-methods
#'@aliases cs_unfold,Tensor-method
setMethod("cs_unfold", signature="Tensor",
definition=function(tnsr,m=NULL){
	if(is.null(m)) stop("mode m must be specified")
	num_modes <- tnsr@num_modes
	rs <- (1:num_modes)[-m]
	cs <- m
	unfold(tnsr,rs=rs,cs=cs)
})
options(warn=1)

###Creation of Tensor from an array/matrix/vector

#'Tensor Conversion
#'
#'Create a \code{\link{Tensor-class}} object from an \code{array}, \code{matrix}, or \code{vector}.
#'@export
#'@name as.tensor
#'@rdname as.tensor
#'@aliases as.tensor
#'@param x an instance of \code{array}, \code{matrix}, or \code{vector}
#'@param drop whether or not modes of 1 should be dropped
#'@return a \code{\link{Tensor-class}} object 
#'@examples
#'#From vector
#'vec <- runif(100); vecT <- as.tensor(vec); vecT
#'#From matrix
#'mat <- matrix(runif(1000),nrow=100,ncol=10)
#'matT <- as.tensor(mat); matT
#'#From array
#'indices <- c(10,20,30,40)
#'arr <- array(runif(prod(indices)), dim = indices)
#'arrT <- as.tensor(arr); arrT
as.tensor <- function(x,drop=FALSE){
	stopifnot(is.array(x)||is.vector(x))
	if (is.vector(x)){
		modes <- c(length(x))
		num_modes <- 1L
	}else{
		modes <- dim(x)
		num_modes <- length(modes)
		dim1s <- which(modes==1)
		if(drop && (length(dim1s)>0)){
			modes <- modes[-dim1s]
			num_modes <- num_modes-length(dim1s)
		}
	}
new("Tensor",num_modes,modes,data=array(x,dim=modes))
}


#####V1.0.1 Addition

#'Image of 3-Tensor Slices
#'
#'Calls \code{image} on the multiple slices of the 3-Tensors to provide more conveinence than calling \code{image} on \code{tnsr@@data}. Need to specify which mode and the range of index along that mode.
#'
#'@docType methods
#'@name image-methods
#'@rdname image-methods
#'@aliases image,Tensor-method
#'@details \code{image(x,m=NULL,indices=NULL,...)}
#'@export
#'@param x the Tensor instance
#'@param m the mode to slice along
#'@param indices the range of indices to be plotted, by default all indices along mode \code{m}
#'@examples
#'tnsr <- rand_tensor(c(3,4,5))
#'#image(tnsr,m=3,indices=c(2:4))
#'#image(tnsr,m=2)
setMethod("image",signature="Tensor",
definition = function(x,m=NULL,indices=NULL,...){
  modes <- x@modes
  numModes <-x@num_modes
  stopifnot((numModes==3)||(numModes==2))
  if(numModes==2){
  	image(x@data,...)
  }
  else{
  stopifnot(m > 0 || m <= 3)
  if(is.null(indices)) indices <- seq(modes[m])
  x <- x@data
  par(mfrow=c(1, length(indices)), mar=c(0.1,0.1,3,0.1))
  if (m==1){
  for (i in 1:length(indices)) image(x[indices[i],,], main=paste(c(indices[i]," of ",modes[m])),...)	
  }else if(m==2){
  for (i in 1:length(indices)) image(x[,indices[i],], main=paste(c(indices[i]," of ",modes[m])),...)  	
  }else{
  for (i in 1:length(indices)) image(x[,,indices[i]], main=paste(c(indices[i]," of ",modes[m])),...)
  }
  }		  	
})

#'Mode Permutation for Tensor
#'
#'Overloads \code{aperm} for Tensor class for convenience. 
#'
#'@docType methods
#'@name tperm-methods
#'@rdname tperm-methods
#'@aliases tperm,Tensor-method
#'@details \code{tperm(x,perm=NULL,...)}
#'@export
#'@param x the Tensor instance
#'@param perm the new permutation of the current modes
#'@examples
#'tnsr <- rand_tensor(c(3,4,5))
#'dim(tperm(tnsr,perm=c(2,1,3)))
#'dim(tperm(tnsr,perm=c(1,3,2)))
setGeneric(name="tperm",
def=function(tnsr,...){standardGeneric("tperm")})

#'@rdname tperm-methods
#'@aliases tperm,Tensor-method
setMethod("tperm",signature="Tensor",
definition=function(tnsr,...){
	as.tensor(aperm(tnsr@data,...))
})

#'@rdname unfold-methods
#'@aliases unfold,Tensor-method vec,Tensor-method
setGeneric(name="vec",def=function(tnsr,...){standardGeneric("vec")})

#'@rdname unfold-methods
#'@aliases unfold,Tensor-method vec,Tensor-method
setMethod("vec",signature="Tensor",
definition=function(tnsr){
	as.tensor(as.vector(tnsr@data))
})

#'@rdname cs_unfold-methods
#'@aliases cs_unfold,Tensor-method matvec,Tensor-method
setGeneric(name="matvec",
           def=function(tnsr,...){standardGeneric("matvec")})

#'@rdname cs_unfold-methods
#'@aliases cs_unfold,Tensor-method matvec,Tensor-method
setMethod('matvec',signature="Tensor",
          definition=function(tnsr){
          num_modes <- tnsr@num_modes
          stopifnot(num_modes==3)
          unfold(tnsr,rs=c(1,3),cs=2)
          })
