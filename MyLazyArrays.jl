module MyLazyArrays

using LinearAlgebra
using LinearAlgebra: checksquare

using LinearAlgebraExtensions: AbstractMatOrFac, AbstractVecOrTup

import Base: getindex, size
import LinearAlgebra: \, /, *, inv, factorize, dot, det

##################### Lazy Multi-Dimensional Grid ##############################
# useful to automatically detect Kronecker structure in Kernel matrices at compile time
struct LazyGrid{T, V<:Tuple{Vararg{AbstractVector{T}}}} <: AbstractVector{AbstractVector{T}}
    args::V
end
Base.length(G::LazyGrid) = prod(length, G.args)
Base.size(G::LazyGrid) = (length(G),)
Base.ndims(G::LazyGrid) = length(G.args) # mh, maybe don't do this?
function Base.getindex(G::LazyGrid{T}, i::Integer) where {T}
    @boundscheck checkbounds(G, i)
    val = zeros(T, ndims(G))
    n = length(G)
    @inbounds for (j, a) in enumerate(G.args)
        n ÷= length(a)
        val[j] = a[cld(i, n)]  # can this be replaced with fld1, mod1?
        i = mod1(i, n) # or some kind of shifted remainder?
    end
    return val
end
grid(args::Tuple{Vararg{AbstractVector}}) = LazyGrid(args)
grid(args::AbstractVector...) = LazyGrid(args)

######################### Lazy Difference Vector ###############################
# TODO: is it worth using the LazyArrays package?
# could be replaced by LazyVector(applied(-, x, y)), which is neat.
# lazy difference between two vectors, has no memory footprint
struct LazyDifference{T, U<:AbstractVecOrTup, V<:AbstractVecOrTup} <: AbstractVector{T} # U, V} <: AbstractVector{T} #
    x::U
    y::V
    function LazyDifference(x, y)
        length(x) == length(y) || throw(DimensionMismatch("x and y do not have the same length: $(length(x)) and $(length(y))."))
        T = promote_type(eltype(x), eltype(y))
        new{T, typeof(x), typeof(y)}(x, y)
    end
end

difference(x::Number, y::Number) = x-y # avoid laziness for scalars
difference(x, y) = LazyDifference(x, y)

size(d::LazyDifference) = (length(d.x),)
getindex(d::LazyDifference, i::Integer) = d.x[i]-d.y[i]
# getindex(d::LazyDifference, ::Colon) = d.x-d.y

# getindex(d::LazyDifference, i::Int) = LazyDifference(x[i], y[i]) # recursive
minus(A::AbstractArray) = ApplyArray(applied(-, A))
# const Minus{T, N} = LazyArray{T, N, typeof(-), ::Tuple{<:AbstractArray{T, N}}}
# const Minus{M<:AbstractArray{T, N}} where {T, N} = ApplyArray{T, N, typeof(-), ::Tuple{M}}

# Minus(A::AbstractArray{T, N}) where {T, N} = Minus{T, N}(A)

############################ Lazy Inverse Matrix ###############################
# converts multiplication into a backsolve
# this makes it very easy to express x' A^{-1} y
# as dot(x, LazyInverse(A), y), without having to form the inverse!
# efficient if A is a Factorization, or a matrix type with non-cubic solves
# e.g. upper-triangular
# TODO: if we don't create different types for Matrix and Factorization,
# it will suffer from the same type inconsistency as Adjoint in LinearAlgebra
# change to Factorization?
# TODO: create custom printing, so that it doesn't fail in the repl
# TODO: could extend Zygote's logdet adjoint with a lazy inverse ...
struct Inverse{T, M} <: Factorization{T} # <: AbstractMatrix{T} ?
    parent::M
    function Inverse(A::M) where {T, M<:AbstractMatOrFac{T}}
        size(A, 1) == size(A, 2) ? new{T, M}(A) : error("Input of size $(size(A)) not square")
    end
end
size(L::Inverse) = size(L.parent)

# Base.show(io::IO, Inv::Inverse) = (println(io, "Inverse of "); show(io, Inv.parent))
function inverse end # smart pseudo-constructor
inverse(Inv::Inverse) = Inv.parent
inv(Inv::Inverse) = inverse(Inv)
inverse(x::Union{Number, Diagonal, UniformScaling}) = inv(x)
inverse(A::AbstractMatOrFac) = Inverse(A)

# factorize the underlying matrix
factorize(Inv::Inverse) = inverse(factorize(Inv.parent))

det(Inv::Inverse) = 1/det(Inv.parent)
LinearAlgebra.logdet(Inv::Inverse) = -logdet(Inv.parent)
dot(x::AbstractVecOrMat, A::Inverse, y::AbstractVecOrMat) = dot(x, A*y)

# TODO:allows for stochastic approximation:
# A Probing Method for Cοmputing the Diagonal of the Matrix Inverse
LinearAlgebra.diag(Inv::Inverse) = diag(Matrix(Inv))
LinearAlgebra.diag(Inv::Inverse{<:Any, <:Factorization}) = diag(inv(Inv.parent))

# TODO: specialize
# mul!(Y, A, B, α, β)
# ldiv!
# rdiv!
import LinearAlgebra: adjoint, transpose, ishermitian, issymmetric
adjoint(Inv::Inverse) = Inverse(adjoint(Inv.parent))
tranpose(Inv::Inverse) = Inverse(tranpose(Inv.parent))
ishermitian(Inv::Inverse) = ishermitian(Inv.parent)
issymmetric(Inv::Inverse) = issymmetric(Inv.parent)
symmetric(Inv::Inverse) = Inverse(Symmetric(Inv.parent))

# TODO: should override factorizations' get factors method instead
import LinearAlgebra: UpperTriangular, LowerTriangular
UpperTriangular(U::Inverse{T, <:UpperTriangular}) where {T} = U
LowerTriangular(L::Inverse{T, <:LowerTriangular}) where {T} = L

# TODO: have to check if these are correct of uplo = L
# inverse(C::Cholesky) = Cholesky(inverse(C.U), C.uplo, C.info)
# inverse(C::CholeskyPivoted) = CholeskyPivoted(inverse(C.U), C.uplo, C.piv, C.rank, C.tol, C.info)

# const Chol = Union{Cholesky, CholeskyPivoted}
# # this should be faster if C is low rank
# *(C::Chol, B::AbstractVector) = C.L * (C.U * B)
# *(C::Chol, B::AbstractMatrix) = C.L * (C.U * B)
# *(B::AbstractVector, C::Chol) = (B * C.L) * C.U
# *(B::AbstractMatrix, C::Chol) = (B * C.L) * C.U

# this implements the right pseudoinverse
# is defined if A has linearly independent columns
# ⁻¹, ⁺ syntax
struct PseudoInverse{T, M<:AbstractMatOrFac{T}} <: Factorization{T}
    parent::M
end
const AbstractInverse{T} = Union{Inverse{T}, PseudoInverse{T}}

Base.size(P::PseudoInverse) = size(P.parent')
Base.size(P::PseudoInverse, k::Integer) = size(P.parent', k::Integer)

function LinearAlgebra.Matrix(P::PseudoInverse)
    A = Matrix(P.parent)
    inverse(A'A)*A' # left pseudo inverse #P.parent'inverse(P.parent*P.parent') # right pseudo inverse
end
LinearAlgebra.AbstractMatrix(A::Adjoint{<:Real, <:PseudoInverse}) = AbstractMatrix(A.parent)'
# function LinearAlgebra.Matrix(A::Adjoint{<:Real, <:PseudoInverse})
#     A = Matrix(A.parent.parent)
#     A*inverse(A*A')
# end
LinearAlgebra.factorize(P::PseudoInverse) = pseudoinverse(factorize(P.parent))

# smart constructor
# TODO: have to figure out how to make right inverse work correctly
# calls regular inverse if matrix is square
function pseudoinverse end
const pinverse = pseudoinverse
function pseudoinverse(A::AbstractMatOrFac, side::Union{Val{:L}, Val{:R}} = Val(:L))
    if size(A, 1) == size(A, 2)
        return inverse(A)
    elseif side isa Val{:L}
        return PseudoInverse(A) # left pinv
    else
        return PseudoInverse(A')' # right pinv
    end
end
pseudoinverse(A::Union{Number, Diagonal, UniformScaling}) = inv(A)
pseudoinverse(P::AbstractInverse) = P.parent

LinearAlgebra.adjoint(P::PseudoInverse) = Adjoint(P)
# LinearAlgebra.transpose(P::PseudoInverse) = Transpose(P)

*(L::AbstractInverse, B::AbstractVector) = L.parent \ B
*(L::AbstractInverse, B::AbstractMatrix) = L.parent \ B

# since left pseudoinverse behaves differently for right multiplication
*(B::AbstractVector, P::PseudoInverse) = B * Matrix(P) #(A = L.parent; (B * inverse(A'A)) * A')
*(B::AbstractMatrix, P::PseudoInverse) = B * Matrix(P) #(A = L.parent; (B * inverse(A'A)) * A')
# *(B::AbstractVector, L::Adjoint{<:Real, <:PseudoInverse}) = B / L.parent
# *(B::AbstractMatrix, L::Adjoint{<:Real, <:PseudoInverse}) = B / L.parent

*(B::AbstractVector, L::Inverse) = B / L.parent
*(B::AbstractMatrix, L::Inverse) = B / L.parent

\(L::AbstractInverse, B::AbstractVector) = L.parent * B
\(L::AbstractInverse, B::AbstractMatrix) = L.parent * B

/(B::AbstractVector, L::AbstractInverse) = B * L.parent
/(B::AbstractMatrix, L::AbstractInverse) = B * L.parent

############################## PeriodicVector ##################################
using InfiniteArrays
# can use this to create an infinite stream of data
struct PeriodicVector{T, V<:AbstractVector{T}} <: AbstractVector{T}
    x::V
end
Base.getindex(P::PeriodicVector, i) = @inbounds P.x[mod1.(i, length(P.x))]
Base.length(::PeriodicVector) = ∞ # TODO: look at InfiniteArrays.jl for how to do this well
Base.size(::PeriodicVector) = (∞,)
Base.firstindex(::PeriodicVector) = -∞
Base.lastindex(::PeriodicVector) = ∞

# maybe not a good idea?
# Base.setindex!(P::PeriodicVector, v, i) = (P.x[mod1(i, length(P.x))] = v)
# TODO: fft? leads to fft(x) padded with infinitely many zeros on each side

##################### Lazy Matrix Sum and Product ##############################
using LazyArrays
function lazysum(A::Union{AbstractMatrix, UniformScaling}...)
    ApplyMatrix(applied(+, A...))
end
lazyprod(A::AbstractMatrix...) = ApplyMatrix(applied(*, A...))

struct LazySum{T, M} <: AbstractMatrix{T}
    args::M
end
function LazySum(args::Tuple{Vararg{Union{AbstractMatrix, UniformScaling}}})
    T = promote_type((eltype(a) for a in args)...)
    LazySum{T, typeof(args)}(args)
end
Base.size(A::LazySum) = size(A.args[1])
Base.getindex(A::LazySum, i, j) = sum(B[i,j] for B in A.args)

struct LazyProduct{T, M} <: AbstractMatrix{T}
    args::M
end
function LazyProduct(args::Tuple{Vararg{Union{AbstractMatrix, UniformScaling}}})
    T = promote_type((eltype(a) for a in args)...)
    LazyProduct{T, typeof(args)}(args)
end
Base.size(A::LazyProduct) = size(A.args[1])
Base.getindex(A::LazyProduct, i, j) = *(A.args[1][i,:], A.args[2:end-1]..., A.args[end][:,j])

# TODO: optimal ordering algorithm
# LinearAlgebra.Matrix(A::LazyProduct) = *(A.args...)

# Matrix multiplication with optimal ordering
# function LinearAlgebra.:*(A::AbstractMatrix...)
#     #, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix...)
#     cost(A) = 0
#     cost(A, B) = prod(size(A)) * size(B, 2) + cost(A) + cost(B)
#     function cost(A...)
#         k = minimum(cost(A[1:i], A[i+1:end]) for i in 1:length(A)-1)
#     end
# end

########################### Hadamard Product ###################################
# TODO: tests and non-allocating versions
struct HadamardProduct{T, A<:Tuple{Vararg{AbstractMatOrFac}}} <: Factorization{T}
    factors::A
    function HadamardProduct(A::Tuple{Vararg{AbstractMatOrFac}})
        T = promote_type(eltype.(A)...)
        all(==(size(A[1]), size.(A))) || error("matrices have to have same size to form Hadamard product")
        HadamardProduct{T, typeof(A)}(A)
    end
end
const SchurProduct = HadamardProduct
# smart constructor
hadamard(A::AbstractMatOrFac...) = HadamardProduct(A)
hadamard(H::HadamardProduct, A::AbstractMatOrFac...) = hadamard(tuple(H.args..., A...))
hadamard(A::AbstractMatOrFac, H::HadamardProduct) = hadamard(tuple(A, H.args...))
const ⊙ = hadamard # \odot

Base.size(H::HadamardProduct) = size(H.factors[1])
Base.getindex(H::HadamardProduct, i, j) = prod(A->A[i, j], H.factors)
Base.eltype(H::HadamardProduct{T}) where {T} = T
function Base.Matrix(H::HadamardProduct)
    [H[i,j] for i in 1:size(H, 1), j in 1:size(H, 2)]
end # could also define AbstractMatrix which could preserve sparsity patterns

# from Horn, Roger A.; Johnson, Charles R. (2012). Matrix analysis. Cambridge University Press.
function Base.:*(H::HadamardProduct, x::AbstractVector)
    checksquare(H) # have to check identity for non-square matrices
    x = Diagonal(x)
    for A in H.factors
        x = (A*x)'
    end
    diag(x)
end
# TODO: check this
inverse(H::HadamardProduct) = hadamard(inverse.(H.factors))
isposdef(H::HadamardProduct) = all(isposdef, H.factors) # this fact makes product kernels p.s.d. (p. 478)
issymmetric(H::HadamardProduct) = all(issymmetric, H.factors)
ishermitian(H::HadamardProduct) = all(ishermitian, H.factors)
issuccess(H::HadamardProduct) = all(issuccess, H.factors)
# TODO: factorize

# function LinearAlgebra.dot(x::AbstractVector, H::Hadamard, y::AbstractVector)
#     tr(Diagonal(x)*H.A*Diagonal(y)*H.B')
# end

#################### Unitary Discrete Fourier Transform ########################
struct ℱ{T} <: AbstractMatrix{T}
    n::Int
end
ℱ(n::Int) = ℱ{Complex{Float64}}(n)

Base.size(F::ℱ) = (F.n, F.n)
Base.getindex(F::ℱ, i, j) = exp(-2π*1im*(i*j)/F.n) / sqrt(F.n)
Base.:*(F::ℱ, A::AbstractArray) = fft(A) / sqrt(F.n) # TODO: check dimensions
Base.:*(A::AbstractArray, F::ℱ) = ifft(A')' / sqrt(F.n) # TODO: check dimensions
Base.:\(F::ℱ, A::AbstractArray) = ifft(A) / sqrt(F.n) # are these necessary?
Base.:/(A::AbstractArray, F::ℱ) = fft(A')' / sqrt(F.n)
Base.inv(F::ℱ) = F' # TODO: make sure adjoints are lazy


end # MyLazyArrays

# first dimension varies most quickly
# function Base.getindex(G::LazyGrid{T}, i::Integer) where {T}
#     @boundscheck checkbounds(G, i)
#     val = zeros(T, ndims(G))
#     n = length(G)
#     j = ndims(G)
#     @inbounds for a in reverse(G.args)
#         n ÷= length(a)
#         val[j] = a[cld(i, n)]  # can this be replaced with fld1, mod1?
#         i = mod1(i, n) # or some kind of shifted remainder?
#         j -= 1
#     end
#     return val
# end
