module SpeedTest
using BenchmarkTools
using KWLinalg
using LinearAlgebra
using Test
using Random

function test_svd_functor(m, n, alg, ele)
    A = rand(MersenneTwister(0), typeof(ele), m, n)
    AC = deepcopy(A)
    svd_functor = svd_functor_divconquer(m, n, Float64)
    U, S, V = svd_functor(A)
    U2, S2, V2 = svd!(AC, alg = alg)
    @test norm(U - U2) == 0.0
    @test norm(S - S2) == 0.0
    @test norm(V - V2) == 0.0
    A = rand(MersenneTwister(0), typeof(ele), m, n)
    svd_alloc = @allocated svd_functor(A)
    # Upon second call, no memory should be allocated.
    println(svd_alloc)
    @test svd_alloc == 0
    return nothing
end

m, n = 5,5
test_svd_functor(m, n, LinearAlgebra.DivideAndConquer(), 1.0)

end
