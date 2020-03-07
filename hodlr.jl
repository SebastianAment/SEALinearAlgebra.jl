using LinearAlgebra

# represents [D1, UV; (UV)', D2]
struct HODLR{T, M, L} <: Factorization{T}
    A::M # diagonal block (could itself be HODLR)
    B::M
    UV::L # off-diagonal, low-rank block, have U, V separate?
    # U::L
    # V::L
end

Base.size(H::HODLR) = size(H.A) .+ size(H.B) # this is the size of the matrix associated with H
LinearAlgebra.issymmetric(::HODLR{Real}) = true
LinearAlgebra.ishermitian(::HODLR) = true
# LinearAlgebra.isposdef(::HODLR) = true # not necessarily true?

# min size is to stop recursion
# TODO: parallelization with tasks
function hodlr(A::AbstractMatrix, min_size::Integer = 32)
    n = size(A, 1)
    if n < min_size
        cholesky(A) # finish diagonal leaf nodes with cholesky
        # return A # or just return dense matrix, if we only need a fast multiply
    else
        i1 = 1:n÷2 # indices for first block
        i2 = n÷2+1:n # indices for second block
        A = hodlr(A[i1, i1]) # should these be views?
        B = hodlr(A[i2, i2])
        UV = lowrank(A[i1, i2])
        HODLR(A, B, UV)
    end
end

# see equation 24 in "Fast Direct Methods for Gaussian Processes"
# recursively converts HODLR factorization into woodbury form
# this is important for inversion and determinant evaluation
# forward application works without converting it to Woodbury form
# TODO: Test Block matrix support
function woodbury(H::HODLR)
    A = Diagonal(woodbury.([H.A, H.B]))
    U = Diagonal([H.U, H.V'])
    V = [Zeros(), H.U'; H.V, Zeros()] # Anti-Diagonal Matrx
    C = I(rank(H.UV)) # could this remain a UniformScaling instance?
    W = Woodbury(A, U, C, V)
end
# The problem with this is that it doesn't preserve the HODLR type
MyLazyArrays.inverse(H::HODLR) = inverse(woodbury(W))

# TODO: non-allocating version
function LinearAlgebra.:*(H::HODLR, X::AbstractMatrix)
    Y = similar(x)
    k = size(H.A, 1)
    Y[1:k, :] .= H.A * X[1:k, :] + H.UV * X[1:k]
    Y[k+1:end, :] .= H.B * X[k+1:end] + H.UV' * X[k+1:end]
    return Y
end
LinearAlgebra.:*(H::HODLR, x::AbstractVector) = reshape(H*reshape(x, (:, 1)), :)

# Sylvester's determinant theorem: det(I + AB) = det(I + BA)
# LinearAlgebra.det(L::LowRank) = det()

LinearAlgebra.det(H::HODLR) = det(woodbury(H))

LinearAlgebra.logdet(H::HODLR) = logdet(woodbury(H))
LinearAlgebra.tr(H::HODLR) = tr(H.A) + tr(H.B)
function LinearAlgebra.:*(A::HODLR, B::HODLR)

end

# not sure if I want to do this ever ...
function Base.Matrix(H::HODLR) end

# function Base.getindex(H::HODLR, i::Integer, j::Integer)
#     if i == 1 && j == 1
#         H.K1
#     elseif i == 2 && j == 1
#         H.UV1
#     elseif i == 1 && j == 2
#         H.UV2
#     elseif i == 2 && j == 2
#         H.K2
#     end
# end
