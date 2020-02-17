module MyFactorizations

using LinearAlgebra
using ..MyLazyArrays: pseudoinverse, PseudoInverse

const AbstractMatOrFac{T} = Union{AbstractMatrix{T}, Factorization{T}}
const AbstractVecOrMatOrFac{T} = Union{AbstractVecOrMat{T}, Factorization{T}}

########################### Projection Matrix ##################################
# stands for A*(AF\y) = A*inverse(A'A)*(A'y) = A*pseudoinverse(A)
struct Projection{T, AT<:AbstractMatOrFac{T},
                                AFT<:AbstractMatOrFac{T}} <: Factorization{T}
    A::AT
    A⁺::AFT
    # temp::V             V<:AbstractVecOrMat
end
Projection(A::AbstractMatOrFac) = Projection(A, pseudoinverse(qr(A, Val(true)))) # defaults to pivoted qr

# TODO: potentially do memory pre-allocation (including temporary)
(P::Projection)(x::AbstractVecOrMatOrFac) = P.A * (P.A⁺ * x)

Base.size(P::Projection, k::Integer) = 0 < k ≤ 2 ? size(P.A, 1) : 1
Base.size(P::Projection) = (size(P, 1), size(P, 2))

# properties
LinearAlgebra.Matrix(P::Projection) = Matrix(P.A * P.A⁺)
LinearAlgebra.adjoint(P::Projection) = P
Base.:^(P::Projection, n::Integer) = P
function Base.literal_pow(::typeof(^), P::Projection, ::Val{N}) where N
    N > 0 ? P : error("Projection P is not invertible")
end
Base.:*(P::Projection, x::AbstractVecOrMat) = P(x)
Base.:*(x::AbstractVecOrMat, P::Projection) = P(x')'

############################# Low Rank #########################################
# Low rank factorization
# A = UV
struct LowRank{T, M<:AbstractMatOrFac, N<:AbstractMatOrFac} <: Factorization{T}
    U::M
    V::N
    rank::Int # rank of the factorization
    tol::T # tolerance / error bound in trace norm
    info::Int # can hold error information about factorization
    function LowRank(U::AbstractMatOrFac, V::AbstractMatOrFac,
                    tol = 1e2eps(eltype(U)), info::Int = 0)
        T = promote_type(eltype(U), eltype(V))
        rank = size(U, 2) == size(V, 1) ? size(U, 2) : throw(
            DimensionMismatch("U and V do not have compatible inner dimensions"))
        new{T, typeof(U), typeof(V)}(U, V, rank, tol, info)
    end
end

function LowRank(U::AbstractMatOrFac, tol::Real = eps(eltype(U)), info::Int = 0)
    LowRank(U, U', tol, info)
end

# should only be used if C.rank < size(C, 1)
function LowRank(C::CholeskyPivoted{T}) where {T}
    ip = invperm(C.p)
    U = C.U[1:C.rank, ip]
    LowRank(U', C.tol, C.info)
end

import LinearAlgebra: issuccess, Matrix, size
issuccess(S::LowRank) = S.info ≥ 0
size(S::LowRank) = (size(S.U, 1), size(S.V, 2))
Matrix(S::LowRank) = S.U*S.V

import LinearAlgebra: dot, *, \, /, adjoint, issymmetric
# adjoint(S::LowRank) = Adjoint(S)
issymmetric(::LowRank) = S.U === S.V'

*(S::LowRank, A::AbstractVecOrMat) = S.U*(S.V*A)
*(A::AbstractVecOrMat, S::LowRank) = (S*A')'

# ternary operations
# we could also write this with a lazy Matrix-Vector product
function dot(X::AbstractVecOrMat, S::LowRank, Y::AbstractVecOrMat)
    UX = U'X
    VY = (issymmetric(S) && X === Y) ? UX : S.V * Y
    dot(XU, VY)
end

function *(X::AbstractVecOrMat, S::LowRank, Y::AbstractVecOrMat)
    XU = X * U
    VY = (issymmetric(S) && X === Y') ? XU' : S.V * Y
    XU*VY
end

# alternating least squares for
function als!(U::AbstractMatrix, V::AbstractMatrix, A::AbstractMatrix,
                tol::Real = 1e-12, maxiter::Int = 32)
    info = -1
    for i in 1:maxiter
        U .= A / V # in place, rdiv!, ldiv!, and qr!, lq!?
        V .= U \ A # need to project rows of V
        if norm(A-U*V) < tol
            info = 0
            break
        end
    end
    return U, V, info
end
als!(L::LowRank, A::AbstractMatrix, maxiter = 32) = als!(L.U, L.V, A, L.tol, maxiter)

# low rank approximation via als
function lowrank(A::AbstractMatrix{T}, rank::Int, tol::Real = 1e-12,
                maxiter::Int = 32) where {T}
    U = rand(eltype(A), (size(A, 1), rank))
    V = rand(eltype(A), rank, size(A, 2))
    U, V, info = als!(U, V, A, tol, maxiter)
    LowRank(U, V, tol, info)
end

# projected alternating least squares
function pals!(L::LowRank, A::AbstractMatrix, PU::Projection, PV::Projection,
                maxiter::Int = 16)
    for i in 1:maxiter
        L.U .= PU(A/L.V) # in place?
        L.V .= PV((L.U\A)')' # need to project rows of V
    end
    return L
end

# U'Ux = A
# # Ux = (U*U') \ (U'A)
# Ux = U' \ A
# x = U \ Ux
# \(S::SymmetricLowRank, A::AbstractVector) = S.U \ (S.U' \ A)
# \(S::SymmetricLowRank, A::AbstractMatrix) = S.U \ (S.U' \ A)
#
# /(A::AbstractVector, S::SymmetricLowRank) = (S \ A')'
# /(A::AbstractMatrix, S::SymmetricLowRank) = (S \ A')'

# least squares for low rank matrices (using pseudo inverse)
# needs testing
# might make sense to let SLR have Factorization field for QR ...
# however, it is more important to make SLR work with Woodbury
# function \(S::SymmetricLowRank, b::AbstractVecOrMat)
#     S.U'S.U x = b
#     # S.U'S.U)'*S.U'S.U x = (S.U'S.U)' b # normal equations
#     # S.U'*(S.U*S.U')*S.U x = (S.U'S.U)*b # normal equations
#     # (S.U*S.U')*S.U x = S.U*b # normal equations has worse conditioning than approach with qr below
#     Ux = S.U' \ b # U*x = U' \ b # has size of rank
#     x = S.U \ Ux # x = U \ (U' \ b) # solves for ls x
# end

# least squares solution
# ldiv!(A::Covariance{T, SymmetricLowRank{T}}, b::AbstractArray{T}) where {T} = ldiv!(A.U', b)

############################ Symmetric Rescaling ###############################
# applications for Schur complement, and VerticalRescaling, SoR
# represents D'*A*D
# could be SymmetricRescaling
struct SymmetricRescaling{T, M<:AbstractMatOrFac{T},
                                    N<:AbstractMatOrFac{T}} <: Factorization{T}
    D::M
    A::N
end
import LinearAlgebra: Matrix, *, \, dot, size
Matrix(L::SymmetricRescaling) = L.D' * L.A * L.D

# doing this:
# applied(*, L.D', L.A, L.D)
# wouldn't be the whole story, because we want to define matrix multiplication
# with SymmetricRescaling, leading to efficiency gains with structure

# avoids forming the full matrix
*(L::SymmetricRescaling, B::AbstractVecOrMat) = L.D'*(L.A*(L.D*B))
\(L::SymmetricRescaling, B::AbstractVecOrMat) = L.D'\(L.A\(L.D\B))

function dot(x::T, L::SymmetricRescaling, y::T) where {T<:AbstractArray}
    dot(L.D*x, L.A, L.D*y)
    # memory allocation can be avoided with lazy arrays:
    # Dx = applied(*, L.D, x)
    # Dy = applied(*, L.D, y)
    # dot(Dx, L.A, Dy)
end

# this might be too general to be efficient for the SymmetricRescaling usecase
# struct LazyMatrixProduct{T, M<:AbstractMatrix{T}} <: AbstractMatrix{T}
#     A::M
#     B::M
# end

######################## Symmetric Low Rank Approximation ######################
# TODO: deprecate in favor of LowRank
# TODO we could move this to a matrix factorization module encompassing NMF etc.
# symmetric low rank matrix factorization
# factorization of the form A = U'U for a symmetric p.s.d. matrix A
# other names: SymPosLowRank, SymmetricOuterProduct
# SymmetricFactorization
# HermitianLowRank, HermitianFactorization
struct SymmetricLowRank{T, M<:AbstractMatOrFac{T}} <: Factorization{T}
    U::M
    rank::Int # rank of the factorization
    tol::T # tolerance / error bound in trace norm
    info::Int # can hold error information about factorization
end
const SLR = SymmetricLowRank
# Constructors
# function SymmetricLowRank{T}(r::Int, n::Int, ε = eps(T)) where {T}
#     SymmetricLowRank(zeros(T, (r, n)), r, ε, 0)
# end
# SymmetricLowRank(r::Int, n::Int, ε::T = eps(Float64)) where {T} = SymmetricLowRank{T}(r, n)
#
function SymmetricLowRank(U::AbstractMatrix{T}, ε::T = eps(Float64)) where {T}
    SymmetricLowRank(U, size(U, 1), ε, 0)
end

# should only be used if C.rank < size(C, 1)
function SymmetricLowRank(C::CholeskyPivoted{T}) where {T}
    ip = invperm(C.p)
    U = C.U[1:C.rank, ip]
    SymmetricLowRank(U, C.rank, C.tol, C.info)
end

import LinearAlgebra: issuccess, Matrix, size
issuccess(S::SymmetricLowRank) = S.info ≥ 0
size(S::SymmetricLowRank) = (size(S.U, 2), size(S.U, 2))
Matrix(S::SymmetricLowRank) = S.U'S.U # U = @view S.U[1:S.rank,:]# S.L * S.L'

import LinearAlgebra: dot, *, \, /, adjoint, issymmetric

adjoint(S::SymmetricLowRank) = S
issymmetric(::SymmetricLowRank) = true

*(S::SymmetricLowRank, A::AbstractVecOrMat) = S.U'*(S.U*A)
*(A::AbstractVecOrMat, S::SymmetricLowRank) = (S*A')'

# U'Ux = A
# # Ux = (U*U') \ (U'A)
# Ux = U' \ A
# x = U \ Ux
\(S::SymmetricLowRank, A::AbstractVector) = S.U \ (S.U' \ A)
\(S::SymmetricLowRank, A::AbstractMatrix) = S.U \ (S.U' \ A)

/(A::AbstractVector, S::SymmetricLowRank) = (S \ A')'
/(A::AbstractMatrix, S::SymmetricLowRank) = (S \ A')'
# this would be a pseudo-inverse ...
# inverse(S::SymmetricLowRank) =

################
# import LinearAlgebra: +
# function +(S::Symmetric, D::Diagonal)
#     T = copy(S)
#     @inbounds for i in 1:size(D, 1)
#         T.data[i, i] += D[i]
#     end
#     return T
# end
function dot(x::AbstractVector, S::SymmetricLowRank, y::AbstractVector)
    # @boundscheck checkbounds()
    e = 0
    if x === y
        for i = 1:size(S.U, 1)
            Ux = 0
            @inbounds @simd for j = 1:size(S.U, 2)
                Ux += (S.U[i,j]*x[j])
            end
            e += Ux^2
        end
    else
        for i = 1:size(S.U, 1)
            Ux = 0
            Uy = 0
            @inbounds @simd for j = 1:size(S.U, 2)
                 Ux += (S.U[i,j]*x[j])
                 Uy += (S.U[i,j]*y[j])
            end
            e += Ux * Uy
        end
    end
    return e
end

end # MyFactorizations
