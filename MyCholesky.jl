module MyCholesky

using MyFactorizations: SymmetricLowRank

################### own implementation of cholesky factoriztions #####################
# why?
# - works for low-rank approximation,
# - works with AbstractMatrix type, which we exploit to not have to construct kernel matrices
# - the above two combined lead to computational savings because of the sparse access pattern
# - in contrast, the LinearAlgebra implementation only works on StridedMatrices
# -> could add generic pivoted cholesky to LinearAlgebra -> make work for complex hermitian matrices
# potentially problematic assumption: LinearAlgebra assumes that setindex! is implemented
# (TODO) parallelize
# TODO: finish making tol a keywork argument throughout
# TODO: have special factorization for Toeplitz matrix
using LinearAlgebra
using LinearAlgebra: Cholesky, CholeskyPivoted
import LinearAlgebra: cholesky, cholesky!, adjoint, dot

# TODO: to check for pointer equality, can get rid of energetic function
# const CholOrPiv{T} = Union{Cholesky{T}, CholeskyPivoted{T}}
# function dot(x::U, C::CholOrPiv, y::V) where {U<:AbstractVecOrMat{<:Real},
#                                             V<:AbstractVecOrMat{<:Real}}
#     if x === y
#         Ux = (C.U * x)
#         return dot(Ux, Ux)
#     else
#         (C.U * y)'(C.U * x)
#     end
#     # (C.U * y)'(C.U * x)
# end

############################### Cholesky #######################################
# non-pivoted cholesky, stores
function cholesky(A::AbstractMatrix{T}, ::Val{false} = Val(false); check::Bool = true) where {T<:Real}
    U = similar(A)
    info = cholesky!(U, A, Val(false); check = check)
    uplo = 'U'
    return Cholesky{eltype(U),typeof(U)}(U, uplo, info)
end

# also works if U = A (overwrites upper triangular part)
function cholesky!(U::AbstractMatrix, A::AbstractMatrix{T},
                                                ::Val{false} = Val(false);
                                                check::Bool = true) where {T<:Real}
    n = LinearAlgebra.checksquare(A)
    n_t = LinearAlgebra.checksquare(U)
    @assert n == n_t # make sure the target matrix is of the same size

    d = diag(A)
    info = 0
    @inbounds for i in 1:n
        if d[i] < 0 # negative pivot
            check && throw(LinearAlgebra.PosDefException(i))
            info = -1
            break
        end
        U[i, i] = sqrt(d[i])

        # TODO: parallelize this loop
        for j = i+1:n # this loop has zero memory allocation!
            dot_mj = zero(T) # dot product
            @simd for k = 1:i-1
                dot_mj += U[k, i] * U[k, j]
            end
            U[i, j] = (A[i, j] - dot_mj)  / U[i, i]
            d[j] -= U[i, j]^2
        end
    end
    return info
end

# (I+K) = (I+U'U)
# (I+U)'(I+U) = I + K + U' + U

############################### Pivoted Cholesky ###############################
# returns PivotedCholesky
function cholesky(A::AbstractMatrix{T}, ::Val{true}; tol::Real = eps(T),
                                            check::Bool = true) where {T<:Real}
    n = LinearAlgebra.checksquare(A)
    U = zero(A)
    piv, rank, ε, info = cholesky!(U, A, Val(true), Val(false), tol; check = check)
    uplo = 'U' # denotes that upper triangular part of A was used to calculate the factorization
    CholeskyPivoted{eltype(U),typeof(U)}(U, uplo, piv, rank, ε, info)
end

# returns SymmetricLowRank
function cholesky(A::AbstractMatrix, ::Val{true}; max_rank::Int,
                                                tol::Real = eps(eltype(A)),
                                                check::Bool = true)
    n = LinearAlgebra.checksquare(A)
    max_rank = min(n, max_rank) # pivoted cholesky terminates after at most n steps
    U = zeros(eltype(A), (max_rank, n))
    piv, rank, tol, info = cholesky!(U, A, Val(true), Val(true), tol; check = check)
    if rank < max_rank # or we allow U to be larger in storage than the rank indicates
        U = U[1:rank, :]
    end
    SymmetricLowRank(U, rank, tol, info)
end

# returns pivots π, rank m, trace norm bound ε, info (0 if successful, -1 if not symmetric, p.s.d., 1 if low rank)
function cholesky!(U::AbstractMatrix{T}, A::AbstractMatrix{T}, ::Val{true},
                        v::V, tol::Real = eps(T);
                        check::Bool = true) where {T<:Real, V<:Union{Val{false}, Val{true}}}
    @assert size(U, 2) == size(A, 2) # outer dimension has to be equal to matrix dimension
    max_iter = size(U, 1)
    U .= zero(T)
    return _chol!(U, A, max_iter, v, tol, check)
end

# returns U s.t. A = U[1:S.rank,:]'U[1:S.rank,:]
# TODO:
# - parallelize using threads and/or tasks
# - test permuting d, so that pivot search is contiguous in mermory,
# - merge the two _chol! methods
# have to check and throw errors for: square, Hermitian, p.s.d
function _chol!(U::M, A::C, max_iter::Int,
                            ::Val{true}, tol::Real = eps(T),
                            check::Bool = true) where {T<:Real, M<:Matrix{T},
                                                C<:AbstractMatrix{T}}
    n = LinearAlgebra.checksquare(A)
    @assert size(U, 1) ≥ max_iter
    @assert size(U, 2) == n
    d = diag(A)
    π = collect(1:n)
    ε = sum(abs, d)
    m = 1
    info = 1 # assuming low-rank, unless algorithm terminates after n steps
    @inbounds while true
        # find pivot element
        max_d = zero(T)
        i = Int(m)
        for k = m:n
            if d[π[k]] > max_d
                max_d = d[π[k]]
                i = k
            end
        end

        if d[π[i]] < T(0) # negative pivot
            m -= 1
            info = -1
            break
        end

        π[i], π[m] = π[m], π[i]
        U[m, π[m]] = sqrt(d[π[m]])

        # Threads.@threads, @inbounds, might need to use Base.@propagate_inbounds
        # can experiment with threads and tasks, spawn one for each j!
        for j = m+1:n # this loop has zero memory allocation! parallelize?
            dot_mj = zero(T) # dot product
            @simd for k = 1:m-1
                dot_mj += U[k, π[m]] * U[k, π[j]]
            end
            U[m, π[j]] = (A[π[m], π[j]] - dot_mj)  / U[m, π[m]]
            d[π[j]] -= U[m, π[j]]^2
        end

        # calculate trace norm error
        ε = zero(T)
        for k = m+1:n
            ε += abs(d[π[k]])
        end

        # termination criterion
        if ε < tol || max_iter ≤ m
            break
        end
        m += 1
    end
    if check && info < 0
        throw(LinearAlgebra.PosDefException(m))
    end
    if m == n # full rank
        info = 0
    end
    return π, m, ε, info
end

# pivoted cholesky which computes upper-triangular U
# returns U s.t. A[π,π] = U'U
# TODO: reorder d according to pivots, allows for simd summation for error bound
function _chol!(U::M, A::C, max_iter::Int,
                                ::Val{false} = Val(false),
                                tol::Real = eps(T),
                                check::Bool = true) where {T<:Real, M<:Matrix{T},
                                                        C<:AbstractMatrix{T}}
    n = LinearAlgebra.checksquare(A)
    n_t = LinearAlgebra.checksquare(U)
    @assert n == n_t # make sure the target matrix is of the same size

    d = diag(A)
    π = collect(1:n)
    ε = sum(abs, d)
    m = 1
    info = 1
    @inbounds while true

        # find pivot element
        max_d = zero(T)
        i = Int(m)
        for k = m:n
            if d[π[k]] > max_d
                max_d = d[π[k]]
                i = k
            end
        end
        # i ≥ m
        if d[π[i]] < T(0) # negative pivot
            m -= 1
            info = -1
            break
        end
        π[i], π[m] = π[m], π[i]

        # swap pivoted column in L
        temp = zero(T)
        for k = 1:n
            temp = U[k, i]
            U[k, i] = U[k, m]
            U[k, m] = temp
        end
        U[m, m] = sqrt(d[π[m]])

        for j = m+1:n # this loop has zero memory allocation!
            dot_mj = zero(T) # dot product
            @simd for k = 1:m-1
                dot_mj += U[k, m] * U[k, j]
            end
            U[m, j] = (A[π[m], π[j]] - dot_mj)  / U[m, m]
            d[π[j]] -= U[m, j]^2
        end

        # calculate trace norm error
        ε = zero(T)
        for k = m+1:n
            ε += abs(d[π[k]])
        end

        # termination criterion
        if ε < tol || max_iter ≤ m
            break
        end
        m += 1
    end
    if check && info < 0
        throw(LinearAlgebra.PosDefException(m))
    end
    if m == n # full rank
        info = 0
    end
    return π, m, ε, info
end

# computes diagonal of matrix from low rank cholesky factorization
# not sure if this is good
import LinearAlgebra: diag
function diag(F::CholeskyPivoted{T}) where {T<:Number}
    n = size(F.L)[1]
    ip = invperm(F.p)
    U = @view F.U[1:F.rank, ip]
    d = zeros(T, n)
    for i = 1:n
        for j = 1:F.rank
            d[i] += U[j,i]^2
        end
    end
    return d
end


##################
# Taken from ToeplitzMatrices,
# TODO: need fast pivoted Bareiss algorithm for low-rank toeplitz matrices
# function cholesky!(L::AbstractMatrix{T}, S::SymmetricToeplitz{T}) where {T<:Number}
#
#     L[:, 1] .= T.vc ./ sqrt(T.vc[1])
#     v = copy(L[:, 1])
#     N = size(T, 1)
#
#     @inbounds for n in 1:N-1
#         sinθn = v[n + 1] / L[n, n]
#         cosθn = sqrt(1 - sinθn^2)
#
#         for n′ in n+1:N
#             v[n′] = (v[n′] - sinθn * L[n′ - 1, n]) / cosθn
#             L[n′, n + 1] = -sinθn * v[n′] + cosθn * L[n′ - 1, n]
#         end
#     end
#     return Cholesky(L, 'L', 0)
# end

end #MyCholesky
