module CovarianceMatrices

using LinearAlgebra
# using ToeplitzMatrices
import LinearAlgebra: factorize

using ..MyCholesky # allows us to compute pivoted cholesky factorization of k(x, x')
# without explicitly forming the kernel matrix matrix

using ..MyFactorizations: SymmetricLowRank

export AbstractCovarianceMatrix
export CovarianceMatrix

const AbstractMatOrFac{T} = Union{AbstractMatrix{T}, Factorization{T}}

# covariance interface
abstract type AbstractCovarianceMatrix{T} <: AbstractMatrix{T} end
# TODO: Should covariance be a trait instead of a wrapper type?

####################### Standard Covariance Matrix #############################
# restrict field to Symmetric, no, but could provide automatic symmetric view!
# should we allow scalars?
struct CovarianceMatrix{T, M<:AbstractMatrix{T}} <: AbstractCovarianceMatrix{T}
    C::M
    function CovarianceMatrix(A::AbstractMatrix; check = true)
        # TODO: how much checking?
        if check
            size(A, 1) == size(A, 2) || throw(DimensionMismatch("CovarianceMatrix has to be square"))
            # issymmetric(A) || error("CovarianceMatrix has to be symmetric")
            # isposdef(A) || error("CovarianceMatrix has to be positive definite")
        end
        new{eltype(A), typeof(A)}(A)
    end
end

# between two distinct vectors or r.v's
struct CrossCovarianceMatrix{T, M<:AbstractMatrix{T}} <: AbstractCovarianceMatrix{T}
    C::M
end

# extending array interface
import Base: size, getindex, setindex!
size(C::AbstractCovarianceMatrix) = size(C.C)
getindex(C::AbstractCovarianceMatrix, i::Int) = C.C[i]
getindex(C::AbstractCovarianceMatrix, i::Int, j::Int) = C.C[i,j]

Covariance(F::Factorization) = CovarianceMatrix(Matrix(F))
# this will become obsolete once my PR goes through
function CovarianceMatrix(F::CholeskyPivoted)
    ip = invperm(F.p)
    U = F.U[1:F.rank, ip]
    CovarianceMatrix(U'U)
end

# properties
LinearAlgebra.issymmetric(A::CovarianceMatrix) = true
LinearAlgebra.isposdef(A::CovarianceMatrix) = true # technically, positive semi-definite, but eh
LinearAlgebra.ishermitian(A::CovarianceMatrix) = true
LinearAlgebra.isposdef(A::CrossCovarianceMatrix) = issymmetric(A) # technically, positive semi-definite, but eh

using ..MyWoodburyMatrices: Woodbury
using ..MyLazyArrays: Inverse
# TODO: think about this
covariance(A::AbstractCovarianceMatrix) = A
covariance(A::AbstractMatOrFac) = issymmetric(A) ? CovarianceMatrix(A) : CrossCovariance(A)
covariance(W::Woodbury) = Woodbury(covariance(W.A), W.U, covariance(W.C), W.V, W.α)
covariance(Inv::Inverse) = Inverse(covariance(Inv.A))

# var(F::CholeskyPivoted{T}) where {T} = diag(F)

# factorize for covariance matrices
# TODO: # potential more specialized types (they are encompassed by allowing the field to be any of these types)
# Toeplitz, HODLR
# new version with my pivoted cholesky
# iscirculant, istoeplitz don't need here, because we will dispatch earlier (see Kernel.jl)
function factorize(C::AbstractCovarianceMatrix; tol = zero(eltype(C)), check = false)
    n = LinearAlgebra.checksquare(C)
    diagC = view(C, diagind(C, 0))
    fudge = 1e2 # TODO: remove fudge
    tol = max(fudge*n*maximum(C)*eps(T), tol)
    if typeof(C) <: CovarianceMatrix
        if typeof(C.C) <: Diagonal
            return C.C
        end
    end
    # could check for bandedness and apply banded Cholesky (MC 160)
    # max_rank = n
    Chol = cholesky(C, Val(true); tol = tol, check = check)
    if Chol.info == 0 # full rank
        return Chol
    elseif Chol.info == 1 # || Chol.tol < tol # low rank
        return SymmetricLowRank(Chol)
    elseif Chol.info == -1 # matrix is not hermitian, p.s.d.
        throw(PosDefException(-1))
        println(Chol.rank)
        println(Chol.tol)
        return Chol
    end
    # cholesky(Matrix(C))
end

# test for covariance function symmetry, positive semi-definiteness, and stationarity.
function iscov(A::M; randomized = false) where {T, M<:AbstractMatrix{T}}
    LinearAlgebra.checksquare(A)
    # symmetry
    # LinearAlgebra.issymmetric(A) tests exact symmetry
    δ = maximum(abs, A - A')
    if δ > 10eps(T)
        # println("Error: Covariance function " * string(K) * " is not symmetric.")
        return false
    end
    Ker = Hermitian(A)

    # positive semi-definiteness: could test via pivoted cholesky
    # tol = 10eps(T)
    # chol = cholesky(A, Val(true), tol, check = false)
    # if chol.info == -1
    #     return false
    # end
    # testing positive semi-definiteness with a relative tolerance based on maximum eigenvalue
    λ = eigvals(A)
    # println(maximum(λ)*eps(T))
    if maximum(λ) < 0 || minimum(λ) < -10maximum(λ)*eps(T)
        return false
    end
    return true
end

end

# function Base.showerror(io::IO, ex::PosSemiDefException)
#     print(io, "PosSemiDefException: matrix is not ")
#     if ex.info == -1
#         print(io, "Hermitian")
#     else
#         print(io, "positive definite")
#     end
#     print(io, "; Cholesky factorization failed.")
# end

# function factorize!(C::Cov, tol::T = 1e-8) where {T<:Real, M<:AbstractMatrix{T},
#                                                 Cov<:Covariance{T, M}}
#
#     n = size(C.C)[1]
#     tol = max(tol, eps(T))
#     if M == Diagonal
#         return C
#     end
#     # elseif M == Circulant
#     #     return C # do fft
#     # end
#
#     Chol = cholesky(C, Val(true), check = false, tol = tol) # pivoted cholesky fallback
#
#     if Chol.info == 0
#         println(Chol)
#         return Chol
#     elseif Chol.rank < n     # if cholesky terminated early
#         rest = @view view(C.C, diagind(C.C, 1))[Chol.rank+1:end]
#         piv = maximum(rest) # pivot value
#         ε = sum(abs, rest) # error bound
#         println(Chol.rank)
#         println(piv)
#         println(ε)
#         println(Chol.info)
#         if piv < zero(T) # the matrix is not p.s.d. if we encounter a negative pivot
#             println("Throw indefinite exception!")
#         elseif ε < tol
#             return Chol
#         end
#     end
# end

# for Cholesky, there are only two error modes: not hermitian, not p.d.
# for pivoted Cholesky, we have that and not p.s.d.
# struct PosSemiDefException <: Exception
#     info::BlasInt
# end
