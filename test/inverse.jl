module TestInverse
using LinearAlgebra
using MyLazyArrays: inverse, Inverse, pseudoinverse, PseudoInverse
using Test

@testset "inverse" begin
    A = randn(3, 3)
    Inv = inverse(A)
    @test A*Inv ≈ I(3)
end

@testset "pseudoinverse" begin
    A = randn(3, 2)
    LInv = pseudoinverse(A)
    @test LInv*A ≈ I(2)
    RInv = pseudoinverse(A, Val(:R))
    @test A*RInv ≈ I(3) # FIXME: have to define matrix multiplication for this 
end


@testset "difference" begin
    using MyLazyArrays: difference, LazyDifference
    n = 3
    x = randn(n)
    y = randn(n)
    d = difference(x, y)
    @test d ≈ x-y
    @test difference(1., 2.) isa Number
end

end # TestInverse
