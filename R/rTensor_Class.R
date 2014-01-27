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
#'    \item{cs_unfold}{\code{signature(tnsr = "Tensor")}: ... }
#'    \item{dim}{\code{signature(tnsr = "Tensor")}: ... }
#'    \item{fnorm}{\code{signature(tnsr = "Tensor")}: ... }
#'    \item{getData}{\code{signature(tnsr = "Tensor")}: ... }
#'    \item{getModes}{\code{signature(tnsr = "Tensor")}: ... }
#'    \item{getNumModes}{\code{signature(tnsr = "Tensor")}: ... }
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
#'    \item{sweep}{\code{signature(tnsr = "Tensor")}: ... }
#'    \item{t}{\code{signature(tnsr = "Tensor")}: ... }
#'    \item{tail}{\code{signature(tnsr = "Tensor")}: ... }
#'    \item{unfold}{\code{signature(tnsr = "Tensor")}: ... }
#'	 }
#'@author James Li \email{jamesyili@@gmail.com}
#'@details {This can be seen as a wrapper class to the base \code{array} class. While it is possible to create an instance using \code{new}, it is also possible to do so by passing the data into \code{\link{as.tensor}}.
#'	
#'All 3 slots of a Tensor instance can be obtained using \code{@@}, but it is recommended to use the getter functions since these have built-in checks. See \code{\link{getModes-methods}}, \code{\link{getNumModes-methods}}, and \code{\link{getData-methods}}.
#'
#'The following methods are overloaded for the Tensor class: \code{\link{dim-methods}}, \code{\link{head-methods}}, \code{\link{tail-methods}}, \code{\link{print-methods}}, \code{\link{show-methods}}, \code{\link{sweep-methods}}, element-wise array operations, and array subsetting.
#'
#'You can always unfold any Tensor into a matrix, and the \code{\link{unfold-methods}}, \code{\link{rs_unfold-methods}}, and \code{\link{cs_unfold-methods}} methods are for that purpose. The output can be kept as a Tensor with 2 modes or a \code{matrix} object. 
#'
#'Conversion from \code{array}/\code{matrix} to Tensor is facilitated via \code{\link{as.tensor}}. To convert from a Tensor instance, simply invoke \code{\link{getData-methods}}.
#'
#'The Frobenius norm of the Tensor is given by \code{\link{fnorm-methods}}, while the inner product between two Tensors (of equal modes) is given by \code{\link{innerProd-methods}}. You can also sum through any one mode to obtain the K-1 Tensor sum using \code{\link{modeSum-methods}}. \code{\link{modeMean-methods}} provides similar functionality to obtain the K-1 Tensor mean. These are primarily meant to be used internally but may be useful in doing statistics with Tensors.
#'
#'For Tensors with 3 modes, we also overloaded \code{t} (transpose), defined by Kilmer et.al (2013). See \code{\link{t-methods}}.
#'
#'To create a Tensor with random entries, see \code{\link{rand_tensor}}.
#'}
#'@note All of the decompositions and regression models in this package require a Tensor input.
#'@references M. Kilmer, K. Braman, N. Hao, and R. Hoover, "Third-order tensors as operators on matrices: a theoretical and computational framework with applications in imaging". SIAM Journal on Matrix Analysis and Applications 2013.
#'@seealso \code{\link{as.tensor}}
#'@examples
#'tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
#'class(tnsr)
#'tnsr
#'print(tnsr)
#'getModes(tnsr)
#'getNumModes(tnsr)
#'head(getData(tnsr))
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

#'Mode Getter for Tensor
#'
#'Return the vector of modes from a tensor. Safer than accessing the slot directly.
#'
#'@docType methods
#'@name getModes-methods
#'@details \code{getModes(tnsr)}
#'@export
#'@rdname getModes-methods
#'@aliases getModes getModes,Tensor-method
#'@param tnsr the Tensor instance
#'@return an integer vector of the modes associated with \code{x}
#'@seealso \code{\link{dim}}
#'@examples
#'tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
#'getModes(tnsr)
setGeneric(name="getModes",
def=function(tnsr){standardGeneric("getModes")})

#'Number of Modes Getter for Tensor
#'
#'Return the number of modes from a tensor. Safer than accessing the slot directly.
#'
#'@docType methods
#'@name getNumModes-methods
#'@details \code{getNumModes(tnsr)}
#'@export
#'@rdname getNumModes-methods
#'@aliases getNumModes getNumModes,Tensor-method
#'@param tnsr the Tensor instance
#'@return an integer of the number of modes associated with \code{x}
#'@examples
#'tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
#'getNumModes(tnsr)
setGeneric(name="getNumModes",
def=function(tnsr){standardGeneric("getNumModes")})

#'Data Getter for Tensor
#'
#'Return the data (\code{array}, \code{matrix}, or \code{vector}) from a tensor. Safer than accessing the slot directly.
#'
#'@docType methods
#'@name getData-methods
#'@details \code{getData(tnsr)}
#'@export
#'@rdname getData-methods
#'@aliases getData getData,Tensor-method
#'@param tnsr the Tensor instance
#'@return a (\code{array}, \code{matrix}, or \code{vector}) that holds the actual data
#'@examples
#'tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
#'getData(tnsr)
setGeneric(name="getData",
def=function(tnsr){standardGeneric("getData")})

#'Tensor Unfolding
#'
#'Unfolds the tensor into a matrix, with the modes in \code{rs} onto the rows and modes in \code{cs} onto the columns. Note that \code{c(rs,cs)} must have the same elements (order doesn't matter) as \code{getModes(x)}. Within the rows and columns, the order of the unfolding is determined by the order of the modes. This convention is consistent with Kolda and Bader (2009). 
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
#'tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
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
#'@references T. Kolda and B. Bader, "Tensor decomposition and applications". SIAM Applied Mathematics and Applications 2009.
#'@param x the Tensor instance
#'@param m the index of the mode to map onto the row space
#'@return atrix with \code{getModes(x)[m]} rows and \code{prod(getModes(x)[-m])} columns
#'@seealso \code{\link{cs_unfold}} and \code{\link{unfold}}
#'@examples
#'tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
#'matT2<-rs_unfold(tnsr,m=2)
setGeneric(name="rs_unfold",
def=function(tnsr,m){standardGeneric("rs_unfold")})

#'Tensor Column Space Unfolding
#'
#'Unfolding of a tensor by mapping the mode 'm' onto the column space, and all other modes onto the row space. For 3-tensors, this is also known as the 'MatVec' operation. This is the prevalent unfolding for TSVD and T-MULT based on block circulant matrices.
#'@docType methods
#'@name cs_unfold-methods
#'@details \code{cs_unfold(tnsr,m=NULL)}
#'@export
#'@rdname cs_unfold-methods
#'@aliases cs_unfold cs_unfold,Tensor-method
#'@references M. Kilmer, K. Braman, N. Hao, and R. Hoover, "Third-order tensors as operators on matrices: a theoretical and computational framework with applications in imaging". SIAM Journal on Matrix Analysis and Applications 2013.
#'@param tnsr the Tensor instance
#'@param m the index of the mode to map onto the column space
#'@return matrix with \code{prod(getModes(x)[-m])} rows and \code{getModes(x)[m]} columns
#'@seealso \code{\link{rs_unfold-methods}} and \code{\link{unfold-methods}}
#'@examples
#'tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
#'matT1<-cs_unfold(tnsr,m=3)
setGeneric(name="cs_unfold",
def=function(tnsr,m){standardGeneric("cs_unfold")})

#'Tensor Sum Across Single Mode
#'
#'Given a mode for a K-tensor, this returns the K-1 tensor resulting from summing across that particular mode.
#'
#'@docType methods
#'@name modeSum-methods
#'@details \code{modeSum(tnsr,m=NULL)}
#'@export
#'@rdname modeSum-methods
#'@aliases modeSum modeSum,Tensor-method
#'@param tnsr the Tensor instance
#'@param m the index of the mode to sum across
#'@return K-1 tensor, where \code{K = getNumModes(x)}
#'@seealso \code{\link{modeMean}} and \code{\link{sweep}}
#'@examples
#'tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
#'modeSum(tnsr,3)
setGeneric(name="modeSum",
def=function(tnsr,m){standardGeneric("modeSum")})

#'Tensor Mean Across Single Mode
#'
#'Given a mode for a K-tensor, this returns the K-1 tensor resulting from taking the mean across that particular mode.
#'
#'@docType methods
#'@name modeMean-methods
#'@details \code{modeMean(tnsr,m=NULL)}
#'@export
#'@rdname modeMean-methods
#'@aliases modeMean modeMean,Tensor-method
#'@param tnsr the Tensor instance
#'@param m the index of the mode to average across
#'@return K-1 Tensor, where \code{K = getNumModes(x)}
#'@seealso \code{\link{modeSum}} and \code{\link{sweep}}
#'@examples
#'tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
#'modeMean(tnsr,1)
setGeneric(name="modeMean",
def=function(tnsr,m){standardGeneric("modeMean")})

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
#'tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
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
#'tnsr1 <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
#'tnsr2 <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
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
#'Return the vector of modes from a tensor. Safer than accessing the slot directly.
#'
#'@name dim-methods
#'@details \code{dim(tnsr)}
#'@export
#'@aliases dim,Tensor-method
#'@docType methods
#'@rdname dim-methods
#'@param tnsr the Tensor instance
#'@return an integer vector of the modes associated with \code{x}
#'@seealso \code{\link{getModes}}
#'@examples
#'tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
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
#'tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
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
#'tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
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
#'tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
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
#'tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
#'tail(tnsr)
setMethod(f="tail",
signature="Tensor",
definition=function(x,...){
	tail(x@data,...)
})

#'Sweep for Tensor
#'
#'Extend sweep for Tensor
#'
#'@name sweep-methods
#'@details \code{sweep(x,m=NULL,stats=NULL,func=NULL,...)}
#'@export
#'@aliases sweep,Tensor-method
#'@docType methods
#'@rdname sweep-methods
#'@param x the Tensor instance
#'@param ... additional parameters to be passed into sweep()
#'@seealso \code{\link{modeSum}} and \code{\link{modeMean}}
#'@examples
#'tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
#'sweep(tnsr,m=c(2,3),stat=1,func='-')
#'sweep(tnsr,m=1,stat=10,func='/')
setMethod("sweep", signature="Tensor",
definition=function(x,m=NULL,stats=NULL,func=NULL,...){
	if(is.null(m)) stop("must specify mode m")
	as.tensor(sweep(x@data,MARGIN=m,STATS=stats,FUN=func,...))
})

#'Extract for Tensor
#'
#'Extends '[' for Tensor class. Works exactly as it would for the base 'array' class.
#'
#'@name [-methods
#'@details \code{x[i,j,...,drop=TRUE]}
#'@export
#'@aliases [,Tensor-method extract,Tensor-method
#'@docType methods
#'@rdname extract-methods
#'@param x Tensor to be subset
#'@param i,j,... indices that specify the extents of the sub-tensor
#'@param drop whether or not to reduce the number of modes to exclude those that have '1' as the mode
#'@return subTensor of class Tensor
#'@examples
#'tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
#'tnsr[1,2,3]
#'tnsr[3,1,]
#'tnsr[,,5]
#'tnsr[,,5,drop=FALSE]
setMethod("[", signature="Tensor",
definition=function(x,i,j,...,drop=TRUE){
	if(!drop) as.tensor(`[`(x@data,i,j,drop=FALSE,...),drop=drop)
	else as.tensor(`[`(x@data,i,j,...))
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
#'tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
#'identical(t(tnsr)@@data[,,1],t(tnsr@@data[,,1]))
#'identical(t(tnsr)@@data[,,2],t(tnsr@@data[,,5]))
#'identical(t(t(tnsr)),tnsr)
setMethod("t",signature="Tensor",
definition=function(tnsr){
	if(tnsr@num_modes!=3) stop("Tensor Transpose currently only implemented for 3d Tensors")
	modes <- tnsr@modes
	new_arr <- array(apply(tnsr@data[,,c(1L,modes[3]:2L)],MARGIN=3,FUN=t),dim=modes[c(2,1,3)])
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
#'tnsr <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
#'tnsr2 <- new("Tensor",3L,c(3L,4L,5L),data=runif(60))
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

#'@rdname getModes-methods
#'@aliases getModes,Tensor-method
setMethod(f="getModes",
signature="Tensor",
definition=function(tnsr){
	tnsr@modes
})

#'@rdname getNumModes-methods
#'@aliases getNumModes,Tensor-method
setMethod(f="getNumModes",
signature="Tensor",
definition=function(tnsr){
	tnsr@num_modes
})

#'@rdname getData-methods
#'@aliases getData,Tensor-method
setMethod(f="getData",
signature="Tensor",
definition=function(tnsr){
	if(tnsr@num_modes==1) return(as.vector(tnsr@data))
	return(tnsr@data)
})

#'@rdname modeSum-methods
#'@aliases modeSum,Tensor-method
setMethod("modeSum",signature="Tensor",
definition=function(tnsr,m=NULL){
	if(is.null(m)) stop("must specify mode m")
	num_modes <- tnsr@num_modes
	if(m<1||m>num_modes) stop("m out of bounds")
	perm <- c(m,(1L:num_modes)[-m])
	arr <- colSums(aperm(tnsr@data,perm),dims=1L)
	as.tensor(arr)
})

#'@rdname modeMean-methods
#'@aliases modeMean,Tensor-method
setMethod("modeMean",signature="Tensor",
definition=function(tnsr,m=NULL){
	if(is.null(m)) stop("must specify mode m")
	num_modes <- tnsr@num_modes
	if(m<1||m>num_modes) stop("m out of bounds")
	perm <- c(m,(1L:num_modes)[-m])
	arr <- colSums(aperm(tnsr@data,perm),dims=1L)
	modes <- tnsr@modes
	as.tensor(arr/modes[m])
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

#'
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
