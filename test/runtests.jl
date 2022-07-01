using LinearAlgebra
using KWLinalg
using Test

@testset "KWLinalg.jl" begin
    for n in [1, 10]
        m = max(n - 2, 1) # Choose rectangular RHS
        A = rand(n, n)
        AC = deepcopy(A)
        A2 = rand(n, n)
        B = rand(n, m)
        X = rand(n, m)
        lu_D = lu!(A2)
        A2 .= AC
        LAPACK.getrf!(lu_D.factors, lu_D.ipiv)
        ldiv_alloc = @allocated ldiv!(X, lu_D, B)
        @test ldiv_alloc == 0
        @test norm(X - A \ B) < 1e-16
        A2 .= AC
        b_alloc = @allocated LAPACK.getrf!(lu_D.factors, lu_D.ipiv)
        # Upon second call, no memory is allocated.
        @test b_alloc == 0
    end
end
