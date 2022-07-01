using LinearAlgebra
using KWLinalg
using Test
using Random

@testset "LU-Updates" begin
    for n in [1, 10]
        m = max(n - 2, 1) # Choose rectangular RHS
        A = rand(MersenneTwister(0), n, n)
        AC = deepcopy(A)
        A2 = rand(MersenneTwister(1), n, n)
        B = rand(MersenneTwister(2), n, m)
        X = rand(MersenneTwister(3), n, m)
        lu_D = lu!(A2)
        A2 .= AC
        LAPACK.getrf!(lu_D.factors, lu_D.ipiv)
        ldiv_alloc = @allocated ldiv!(X, lu_D, B)
        @test ldiv_alloc == 0
        @test norm(X - A \ B) < 1e-16
        A2 .= AC
        getrf_alloc = @allocated LAPACK.getrf!(lu_D.factors, lu_D.ipiv)
        # Upon second call, no memory should be allocated.
        @test getrf_alloc == 0
    end
end

function test_complex_svd_functor(m, n, fun, alg)
    A = rand(MersenneTwister(0), ComplexF64, m, n)
    AC = deepcopy(A)
    svd_functor = fun(m, n)
    U, S, V = svd_functor(A)
    U2, S2, V2 = svd!(AC, alg = alg)
    @test norm(U - U2) == 0.0
    @test norm(S - S2) == 0.0
    @test norm(V - V2) == 0.0
    A = rand(MersenneTwister(0), ComplexF64, m, n)
    svd_alloc = @allocated svd_functor(A)
    # Upon second call, no memory should be allocated.
    @test svd_alloc == 0
end

@testset "Complex SVD" begin
    for n in [1, 10]
        for m in [1, 10]
            for (fun, alg) in [
                (complex_svd_divconquer!, LinearAlgebra.DivideAndConquer()),
                (complex_svd_qr!, LinearAlgebra.QRIteration()),
            ]
                test_complex_svd_functor(n, m, fun, alg)
            end
        end
    end
end
