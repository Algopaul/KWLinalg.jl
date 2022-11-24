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

function test_svd_functor(m, n, fun, alg, ele, full)
    A = rand(MersenneTwister(0), typeof(ele), m, n)
    AC = deepcopy(A)
    svd_functor = fun(m, n, typeof(ele); full_description(fun, full)...)
    U, S, V = svd_functor(A)
    U2, S2, V2 = svd!(AC, alg = alg, full = full)
    @test norm(U - U2) == 0.0
    @test norm(S - S2) == 0.0
    @test norm(V - V2) == 0.0
    A = rand(MersenneTwister(0), typeof(ele), m, n)
    svd_alloc = @allocated svd_functor(A)
    # Upon second call, no memory should be allocated.
    @test svd_alloc == 0
    return nothing
end

function test_svd_with_smaller_matrix(dtype)
    f1 = svd_functor_divconquer(10, 5, dtype)
    f2 = svd_functor_qr(10, 5, dtype)
    A = rand(MersenneTwister(0), dtype, 8, 4)
    U1, S1, V1 = f1(deepcopy(A))
    U2, S2, V2 = f2(deepcopy(A))
    U1r, S1r, V1r = svd(A)
    U2r, S2r, V2r = svd(A, alg=LinearAlgebra.QRIteration())
    @test norm(U1 - U1r) == 0
    @test norm(S1 - S1r) == 0
    @test norm(V1 - V1r) == 0
    @test norm(U2 - U2r) < 10*eps(dtype)
    @test norm(S2 - S2r) < 10*eps(dtype)
    @test norm(V2 - V2r) < 10*eps(dtype)
end

function full_description(fun, full)
    if fun == svd_functor_divconquer
        if full == true
            return (JOBZ=Cchar('A'),)
        else
            return (JOBZ=Cchar('S'),)
        end
    elseif fun == svd_functor_qr
        if full == true
            return (JOBU=Cchar('A'), JOBVT=Cchar('A'))
        else
            return (JOBU=Cchar('S'), JOBVT=Cchar('S'))
        end
    else
        @error "Unknown function name"
    end
end

@testset "Complex SVD" begin
    @testset "Functionaliy and Allocs" begin
        for n in [1, 10]
            for m in [1, 10]
                for dtype in [Float32, Float64, ComplexF32, ComplexF64]
                    for full in [false, true]
                        test_svd_functor(
                            m,
                            n,
                            svd_functor_divconquer,
                            LinearAlgebra.DivideAndConquer(),
                            one(dtype),
                            full
                        )
                        test_svd_functor(
                            m,
                            n,
                            svd_functor_qr,
                            LinearAlgebra.QRIteration(),
                            one(dtype),
                            full
                        )
                    end
                end
                for dtype in [Float32, Float64]
                    test_svd_with_smaller_matrix(dtype)
                end
            end
        end
    end

    @testset "Exceptions" begin
        @test_throws ErrorException("unsupported dtype") svd_functor_divconquer(1, 1, Int)
        @test_throws ErrorException("unsupported dtype") svd_functor_qr(1, 1, Int)
        @test_throws ErrorException("JOBU must be a Cchar with value 'A' or 'S'") KWLinalg.get_uvt(
            Cchar('B'),
            Cchar('A'),
            3,
            3;
            dtype = ComplexF64,
        )
        @test_throws ErrorException("JOBVT must be a Cchar with value 'A' or 'S'") KWLinalg.get_uvt(
            Cchar('A'),
            Cchar('B'),
            3,
            3;
            dtype = ComplexF64,
        )
    end

end
