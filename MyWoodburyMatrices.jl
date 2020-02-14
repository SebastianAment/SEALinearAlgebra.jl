module MyWoodburyMatrices
using LinearAlgebra
using LinearAlgebra: checksquare

# using LazyArrays: applied, ApplyMatrix
import MyLazyArrays: inverse, Inverse
import LinearAlgebra: factorize, dot, *, Symmetric, Hermitian, \, /

const AbstractMatOrFac{T} = Union{AbstractMatrix{T}, Factorization{T}}
# Un = Union{AbstractMatrix, Factorization}

# TODO: pre-allocate intermediate storage

# represents A + αUCV
# the α is beneficial to preserve p.s.d.-ness during inversion (see inverse)
struct Woodbury{T, AT, UT, CT, VT} <: Factorization{T}
    A::AT
    U::UT
    C::CT
    V::VT
	α::T
	function Woodbury(A::AbstractMatOrFac, U::AbstractMatOrFac,
						C::AbstractMatOrFac, V::AbstractMatOrFac, α::Number = 1)
		checkdims(A, U, C, V)
		# check promote_type
		T = promote_type(eltype.((A, U, C, V))...)
		α = convert(T, α)
		new{T, typeof(A), typeof(U), typeof(C), typeof(V)}(A, U, C, V, α)
	end
end

# checks if the dimensions of A, U, C, V are consistent to form a Woodbury factorization
function checkdims(A, U, C, V)
	n = checksquare(A)
	k = checksquare(C)
	s = "is inconsistent with A ($(size(A))) and C ($(size(C)))"
	size(U) ≠ (n, k) && throw(DimensionMismatch("Size of U ($(size(U)))"*s))
	size(V) ≠ (k, n) && throw(DimensionMismatch("Size of V ($(size(V)))"*s))
	return
end

# function Woodbury(A::Union{AbstractMatrix, Factorization}, U::AbstractMatrix,
# 				C::Union{AbstractMatrix, Factorization},
# 				V::AbstractMatrix)#, α = 1)
# 				println("hi")
# 	Woodbury{T, typeof(A), typeof(U), typeof(C), typeof(V)}(A, U, C, V, convert(T, 1))
# end

# remnants
# Cp = inv(Matrix(inv(C) .+ V*(A\U)))
# temporary space for allocation-free solver
# tmpN1 = Array{T,1}(undef, N)
# tmpN2 = Array{T,1}(undef, N)
# tmpk1 = Array{T,1}(undef, k)
# tmpk2 = Array{T,1}(undef, k)

# TODO: make it work if U, V are vectors
# original : Vector , Matrix
# function Woodbury(A, U::Vector, C, V::Adjoint{<:Number, Vector}, α = 1)
# 	Woodbury(A, reshape(U, , C, V, α)
# end
# Woodbury(A, U::AbstractVector, C, V::Adjoint) = Woodbury(A, U, C, Matrix(V))

LinearAlgebra.size(W::Woodbury) = size(W.A)
LinearAlgebra.size(W::Woodbury, d) = size(W.A, d)

Base.eltype(W::Woodbury{T}) where {T} = T
LinearAlgebra.issymmetric(W::Woodbury) = eltype(W) <: Real && ishermitian(W)
LinearAlgebra.ishermitian(W::Woodbury) = W.U ≡ W.V' && ishermitian(W.A) && ishermitian(W.C)
LinearAlgebra.Matrix(W::Woodbury) = W.A + W.α * *(W.U, W.C, W.V)

function Base.deepcopy(W::Woodbury)
	U = deepcopy(W.U)
	V = U ≡ V' ? U' : deepcopy(V)
	Woodbury(deepcopy(W.A), U, deepcopy(W.C), V, W.α)
end

# TODO: take care of temporaries
*(W::Woodbury, B::AbstractVecOrMat) = W.A*B + W.α*W.U*(W.C*(W.V*B))
*(B::AbstractMatrix, W::Woodbury) = (W'*B')'

function LinearAlgebra.adjoint(W::Woodbury)
	ishermitian(W) ? W : Woodbury(W.A', W.V', W.C', W.U', conj(W.α))
end

function LinearAlgebra.transpose(W::Woodbury)
	issymmetric(W) ? W : Woodbury(transpose.((W.A, W.V, W.C, W.U)), W.α)
end

\(W::Woodbury, B::AbstractVecOrMat) = inverse(W)*B
/(B::AbstractMatrix, W::Woodbury) = B*inverse(W)

# TODO: check this and make it efficient
function LinearAlgebra.det(W::Woodbury)
	det(W.A) * det(W.C) * det(inv(W.C) + *(W.V, inverse(W.A), W.U))
end

# TODO: implement this in WoodburyMatrices and PR
# figure out constant c for which woodbury is most efficient
function inverse(W::Woodbury, c::Real = 1)
	if size(W.U, 1) > c*size(W.U, 2) # and how easy is it to invert W.A, and W.C
		invA = inv(W.A)
		AU = invA * W.U
		VA = (W.U ≡ W.V' && ishermitian(invA)) ? AU' : W.V * invA
        C = inverse(inv(W.C) + W.V * AU) # could be made more efficient with 5 arg mul
		Woodbury(invA, AU, C, VA, -W.α)
    else
        Inverse(W)
    end
end

function LinearAlgebra.factorize(W::Woodbury, c::Real = 1)
    if size(W.U, 1) > c*size(W.U, 2) # only use Woodbury identiy when it is beneficial to do so
		W = inverse(W) # triggers woodbury identiy
        W = Woodbury(factorize(W.A), W.U, factorize(W.C), W.V, W.α)
		Inverse(W) # wraps in Lazy inverse
    else
        factorize(Matrix(W))
    end
end

# TODO: remove for centralized definition
dot(x, A, y) = dot(x, A*y)

# ternary dot
function LinearAlgebra.dot(x::AbstractVecOrMat, W::Woodbury, y::AbstractVecOrMat)
    Ux = W.U'x     # memory allocation can be avoided with lazy arrays
    Vy = (x ≡ y && W.U ≡ W.V') ? Ux : W.V*y
    dot(x, W.A, y) + W.α * dot(Ux, W.C, Vy)
end

# ternary mul
function *(x::AbstractMatrix, W::Woodbury, y::AbstractVecOrMat)
    xU = x*W.U
    Vy = (x ≡ y' && W.U ≡ W.V') ? xU' : W.V*y
    *(x, W.A, y) + W.α * *(xU, W.C, Vy) # can avoid two temporaries
end

end # MyWoodburyMatrices

# function show(io::IO, W::Woodbury)
#     println(io, "Woodbury factorization:\nA:")
#     show(io, MIME("text/plain"), W.A)
#     print(io, "\nU:\n")
#     Base.print_matrix(IOContext(io, :compact=>true), W.U)
#     if isa(W.C, Matrix)
#         print(io, "\nC:\n")
#         Base.print_matrix(IOContext(io, :compact=>true), W.C)
#     else
#         print(io, "\nC: ", W.C)
#     end
#     print(io, "\nV:\n")
#     Base.print_matrix(IOContext(io, :compact=>true), W.V)
# end

# TODO:
# function ldiv!(W::Woodbury, B::AbstractVector)
#     length(B) == size(W, 1) || throw(DimensionMismatch("Vector length $(length(B)) must match matrix size $(size(W,1))"))
#     copyto!(W.tmpN1, B)
#     Alu = lu(W.A) # Note. This makes an allocation (unless A::LU). Alternative is to destroy W.A.
#     ldiv!(Alu, W.tmpN1)
#     mul!(W.tmpk1, W.V, W.tmpN1)
#     mul!(W.tmpk2, W.Cp, W.tmpk1)
#     mul!(W.tmpN2, W.U, W.tmpk2)
#     ldiv!(Alu, W.tmpN2)
#     for i = 1:length(W.tmpN2)
#         @inbounds B[i] = W.tmpN1[i] - W.tmpN2[i]
#     end
#     B
# end

# Base.getindex(W::SymmetricWoodbury, i::UnitRange, j::UnitRange) =
#   i ≡ j ? SymmetricWoodbury(W.A[i,i], W.B[i,:], W.D) : Matrix(W)[i,j]

# function woodbury(A, U, C, V, α = 1, c::Real = 1)
#     W = Woodbury(A, U, C, V, α)
#     size(W.U, 1) > c * size(W.U, 2) ? W : Matrix(W) # only return Woodbury if it is efficient
# end
