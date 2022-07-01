module KWLinalg

using LinearAlgebra

import LinearAlgebra:
    require_one_based_indexing,
    chkstride1,
    BlasInt,
    LAPACK.chkargsok,
    LAPACK.getrf!,
    LinearAlgebra.BLAS.libblastrampoline,
    LinearAlgebra.BLAS.@blasfunc,
    svd!

for (getrf, dtype) in [(:dgetrf_, Float64), (:zgetrf_, ComplexF64)]
    @eval begin
        """
            getrf!(A, ipiv) -> (A, ipiv, info)

        Compute the LU factorization of a general M-by-N matrix `A`.
        The pivot-vector `ipiv` can be provided to avoid an allocation
        """
        function getrf!(A::AbstractMatrix{$dtype}, ipiv::Vector{BlasInt})
            require_one_based_indexing(A)
            chkstride1(A)
            m, n = size(A)
            @assert length(ipiv) >= min(m, n)
            lda = max(1, stride(A, 2))
            info = Ref{BlasInt}()
            ccall(
                (LinearAlgebra.BLAS.@blasfunc($getrf), libblastrampoline),
                Cvoid,
                (
                    Ref{BlasInt},
                    Ref{BlasInt},
                    Ptr{ComplexF64},
                    Ref{BlasInt},
                    Ptr{BlasInt},
                    Ptr{BlasInt},
                ),
                m,
                n,
                A,
                lda,
                ipiv,
                info,
            )
            chkargsok(info[])
            return A, ipiv, info[]
        end
    end
end

gesv = :zgesvd_
@eval begin
    function complex_svd_qr!(
        A::AbstractMatrix{ComplexF64};
        JOBU::Cchar = Cchar('A'),
        JOBVT::Cchar = Cchar('A'),
        M::BlasInt = size(A, 1),
        N::BlasInt = size(A, 2),
        LDA::BlasInt = max(1, stride(A, 2)),
        S::AbstractVector{Float64} = Vector{Float64}(undef, min(size(A)...)),
        U::AbstractMatrix{ComplexF64} = Matrix{ComplexF64}(undef, size(A, 1), size(A, 1)),
        LDU::BlasInt = max(1, stride(U, 2)),
        VT::AbstractMatrix{ComplexF64} = Matrix{ComplexF64}(undef, size(A, 2), size(A, 2)),
        LDVT::BlasInt = max(1, stride(VT, 2)),
        LWORK::BlasInt = 10 * max(1, 2 * min(M, N) + max(M, N)),
        WORK::Vector{ComplexF64} = Vector{ComplexF64}(undef, LWORK),
        RWORK::Vector{Float64} = Vector{Float64}(undef, 5 * min(M, N)),
        INFO = 0,
    )
        ccall(
            (@blasfunc($gesv), libblastrampoline),
            Cvoid,
            (
                Ref{Int8},       # JOBU
                Ref{Int8},       # JOBVT
                Ref{Int64},      # M
                Ref{Int64},      # N
                Ptr{ComplexF64}, # A
                Ref{Int64},      # LDA
                Ptr{Float64},    # S
                Ptr{ComplexF64}, # U
                Ref{Int64},      # LDU
                Ptr{ComplexF64}, # VT
                Ref{Int64},      # LDVT
                Ptr{ComplexF64}, # WORK
                Ref{Int64},      # LWORK
                Ptr{Float64},    # RWORK
                Ref{Int64},      # INFO
            ),
            JOBU,
            JOBVT,
            M,
            N,
            A,
            LDA,
            S,
            U,
            LDU,
            VT,
            LDVT,
            WORK,
            LWORK,
            RWORK,
            INFO,
        )
        return SVD(U, S, VT)
    end
end

function complex_svd_qr!(M, N; JOBU = Cchar('S'), JOBVT = Cchar('S'))
    LDA = max(1, M)
    S = Vector{Float64}(undef, min(M, N))
    U, VT = get_uvt(JOBU, JOBVT, M, N)
    LDU = max(1, stride(U, 2))
    LDVT = max(1, stride(VT, 2))
    LWORK = 10 * max(1, 2 * min(M, N) + max(M, N))
    WORK = Vector{ComplexF64}(undef, LWORK)
    RWORK = Vector{Float64}(undef, 5 * min(M, N))
    INFO = 0
    return A -> complex_svd_qr!(A; JOBU, JOBVT, M, N, LDA, S, U, LDU, VT, LDVT, LWORK, WORK, RWORK, INFO)
end

gesd = :zgesdd_
@eval begin
    function complex_svd_divconquer!(
        A::AbstractMatrix{ComplexF64};
        JOBZ::Cchar = Cchar('A'),
        M::BlasInt = size(A, 1),
        N::BlasInt = size(A, 2),
        LDA::BlasInt = max(1, stride(A, 2)),
        S::AbstractVector{Float64} = Vector{Float64}(undef, min(size(A)...)),
        U::AbstractMatrix{ComplexF64} = Matrix{ComplexF64}(undef, size(A, 1), size(A, 1)),
        LDU::BlasInt = max(1, stride(U, 2)),
        VT::AbstractMatrix{ComplexF64} = Matrix{ComplexF64}(undef, size(A, 2), size(A, 2)),
        LDVT::BlasInt = max(1, stride(VT, 2)),
        LWORK::BlasInt = 2 * min(M, N) * min(M, N) + 2 * min(M, N) + max(M, N),
        WORK::Vector{ComplexF64} = Vector{ComplexF64}(
            undef,
            10 * max(1, 2 * min(M, N) + max(M, N)),
        ),
        LRWORK::BlasInt = max(
            5 * min(M, N) * min(M, N) + 4 * min(M, N),
            2 * max(M, N) * min(M, N) + 2 * min(M, N) * min(M, N) + min(M, N),
        ),
        RWORK::Vector{Float64} = Vector{Float64}(undef, LRWORK),
        IWORK::Vector{BlasInt} = Vector{BlasInt}(undef, 8 * min(M, N)),
        INFO = 0,
    )
        ccall(
            (@blasfunc($gesd), libblastrampoline),
            Cvoid,
            (
                Ref{Int8},   # JOBZ
                Ref{Int64}, # M
                Ref{Int64}, # N
                Ptr{ComplexF64}, # A
                Ref{Int64},   # LDA
                Ptr{Float64}, # S
                Ptr{ComplexF64}, # U
                Ref{Int64},   # LDU
                Ptr{ComplexF64}, # VT
                Ref{Int64},   # LDVT
                Ptr{ComplexF64}, # WORK
                Ref{Int64},   # LWORK
                Ptr{Float64}, # RWORK
                Ptr{Int64}, # RWORK
                Ref{Int64}, # INFO
            ),
            JOBZ,
            M,
            N,
            A,
            LDA,
            S,
            U,
            LDU,
            VT,
            LDVT,
            WORK,
            LWORK,
            RWORK,
            IWORK,
            INFO,
        )
        return SVD(U, S, VT)
    end
end

function complex_svd_divconquer!(M, N; JOBZ = Cchar('S'))
    LDA = max(1, M)
    S = Vector{Float64}(undef, min(N, M))
    U, VT = get_uvt(JOBZ, JOBZ, M, N)
    LDU = max(1, stride(U, 2))
    LDVT = max(1, stride(VT, 2))
    LWORK = 2 * min(M, N) * min(M, N) + 2 * min(M, N) + max(M, N)
    WORK = Vector{ComplexF64}(undef, 10 * max(1, 2 * min(M, N) + max(M, N)))
    LRWORK = max(
        5 * min(M, N) * min(M, N) + 4 * min(M, N),
        2 * max(M, N) * min(M, N) + 2 * min(M, N) * min(M, N) + min(M, N),
    )
    RWORK = Vector{Float64}(undef, LRWORK)
    IWORK = Vector{BlasInt}(undef, 8 * min(M, N))
    INFO = 0
    return A -> complex_svd_divconquer!(
        A;
        JOBZ,
        LDA,
        S,
        U,
        LDU,
        VT,
        LDVT,
        LWORK,
        WORK,
        LRWORK,
        RWORK,
        IWORK,
        INFO,
    )
end

function get_uvt(JOBU, JOBVT, M, N)
    if JOBU == Cchar('A')
        U = Matrix{ComplexF64}(undef, M, M)
    elseif JOBU == Cchar('S')
        U = Matrix{ComplexF64}(undef, M, min(M, N))
    else
        error("JOBU must be 'A' or 'S'")
    end
    if JOBVT == Cchar('A')
        VT = Matrix{ComplexF64}(undef, N, N)
    elseif JOBVT == Cchar('S')
        VT = Matrix{ComplexF64}(undef, min(M, N), N)
    else
        error("JOBVT must be a Cchar with value 'A' or 'S'")
    end
    return U, VT
end


export complex_svd_qr!, complex_svd_divconquer!

end

