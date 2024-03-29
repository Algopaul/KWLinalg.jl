for (gesv, dtype) in [(:zgesvd_, ComplexF64), (:cgesvd_, ComplexF32)]
    @eval begin
        function svd_qr!(
            A::AbstractMatrix{$dtype};
            JOBU::Cchar = Cchar('A'),
            JOBVT::Cchar = Cchar('A'),
            M::BlasInt = size(A, 1),
            N::BlasInt = size(A, 2),
            LDA::BlasInt = max(1, stride(A, 2)),
            S::AbstractVector{$(real(dtype))} = Vector{$(real(dtype))}(undef, min(M, N)),
            U::AbstractMatrix{$dtype} = Matrix{$dtype}(undef, size(A, 1), size(A, 1)),
            LDU::BlasInt = max(1, stride(U, 2)),
            VT::AbstractMatrix{$dtype} = Matrix{$dtype}(undef, size(A, 2), size(A, 2)),
            LDVT::BlasInt = max(1, stride(VT, 2)),
            LWORK::BlasInt = 10 * max(1, 2 * min(M, N) + max(M, N)),
            WORK::Vector{$dtype} = Vector{$dtype}(undef, LWORK),
            RWORK::Vector{$(real(dtype))} = Vector{$(real(dtype))}(undef, 5 * min(M, N)),
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
                    Ptr{$dtype}, # A
                    Ref{Int64},      # LDA
                    Ptr{$(real(dtype))},    # S
                    Ptr{$dtype}, # U
                    Ref{Int64},      # LDU
                    Ptr{$dtype}, # VT
                    Ref{Int64},      # LDVT
                    Ptr{$dtype}, # WORK
                    Ref{Int64},      # LWORK
                    Ptr{$(real(dtype))},    # RWORK
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
end

function get_lwork_gesvd_real(M, N)
    return max(1, 3 * min(M, N) + max(M, N), 5 * min(M, N))
end

for (gesv, dtype) in [(:dgesvd_, Float64), (:sgesvd_, Float32)]
    @eval begin
        function svd_qr!(
            A::AbstractMatrix{$dtype};
            JOBU::Cchar = Cchar('A'),
            JOBVT::Cchar = Cchar('A'),
            M::BlasInt = size(A, 1),
            N::BlasInt = size(A, 2),
            LDA::BlasInt = max(1, stride(A, 2)),
            S::AbstractVector{$dtype} = Vector{$dtype}(undef, min(M, N)),
            U::AbstractMatrix{$dtype} = Matrix{$dtype}(undef, size(A, 1), size(A, 1)),
            LDU::BlasInt = max(1, stride(U, 2)),
            VT::AbstractMatrix{$dtype} = Matrix{$dtype}(undef, size(A, 2), size(A, 2)),
            LDVT::BlasInt = max(1, stride(VT, 2)),
            LWORK::BlasInt = get_lwork_gesvd_real(M, N),
            WORK::Vector{$dtype} = Vector{$dtype}(undef, LWORK),
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
                    Ptr{$dtype}, # A
                    Ref{Int64},      # LDA
                    Ptr{$(real(dtype))},    # S
                    Ptr{$dtype}, # U
                    Ref{Int64},      # LDU
                    Ptr{$dtype}, # VT
                    Ref{Int64},      # LDVT
                    Ptr{$dtype}, # WORK
                    Ref{Int64},      # LWORK
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
                INFO,
            )
            return SVD(U, S, VT)
        end
    end
end

function get_lwork_gesdd_real(M, N, JOBZ)
    mx = max(M, N)
    mn = min(M, N)
    if JOBZ == Cchar('N')
        return 3 * mn + max(mx, 7 * mn)
    elseif JOBZ == Cchar('O')
        return 3 * mn + max(mx, 5 * mn * mn + 4 * mn)
    elseif JOBZ == Cchar('S')
        return 4 * mn * mn + 7 * mn
    elseif JOBZ == Cchar('A')
        return 4 * mn * mn + 6 * mn + mx
    end
end

for (gesd, dtype) in [(:dgesdd_, Float64), (:sgesdd_, Float32)]
    @eval begin
        function svd_divconquer!(
            A::AbstractMatrix{$dtype};
            JOBZ::Cchar = Cchar('A'),
            M::BlasInt = size(A, 1),
            N::BlasInt = size(A, 2),
            LDA::BlasInt = max(1, stride(A, 2)),
            S::AbstractVector{$(real(dtype))} = Vector{$(real(dtype))}(undef, min(M, N)),
            U::AbstractMatrix{$dtype} = Matrix{$dtype}(undef, size(A, 1), size(A, 1)),
            LDU::BlasInt = max(1, stride(U, 2)),
            VT::AbstractMatrix{$dtype} = Matrix{$dtype}(undef, size(A, 2), size(A, 2)),
            LDVT::BlasInt = max(1, stride(VT, 2)),
            LWORK::BlasInt = get_lwork_gesdd_real(M, N, JOBZ),
            WORK::Vector{$dtype} = Vector{$dtype}(undef, LWORK),
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
                    Ptr{$dtype}, # A
                    Ref{Int64},   # LDA
                    Ptr{$(real(dtype))}, # S
                    Ptr{$dtype}, # U
                    Ref{Int64},   # LDU
                    Ptr{$dtype}, # VT
                    Ref{Int64},   # LDVT
                    Ptr{$dtype}, # WORK
                    Ref{Int64},   # LWORK
                    Ptr{Int64}, # IWORK
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
                IWORK,
                INFO,
            )
            return SVD(U, S, VT)
        end
    end
end

for (gesd, dtype) in [(:zgesdd_, ComplexF64), (:cgesdd_, ComplexF32)]
    @eval begin
        function svd_divconquer!(
            A::AbstractMatrix{$dtype};
            JOBZ::Cchar = Cchar('A'),
            M::BlasInt = size(A, 1),
            N::BlasInt = size(A, 2),
            LDA::BlasInt = max(1, stride(A, 2)),
            S::AbstractVector{$(real(dtype))} = Vector{$(real(dtype))}(undef, min(M, N)),
            U::AbstractMatrix{$dtype} = Matrix{$dtype}(undef, M, M),
            LDU::BlasInt = max(1, stride(U, 2)),
            VT::AbstractMatrix{$dtype} = Matrix{$dtype}(undef, N, N),
            LDVT::BlasInt = max(1, stride(VT, 2)),
            LWORK::BlasInt = 2 * min(M, N) * min(M, N) + 2 * min(M, N) + max(M, N),
            WORK::Vector{$dtype} = Vector{$dtype}(
                undef,
                10 * max(1, 2 * min(M, N) + max(M, N)),
            ),
            LRWORK::BlasInt = max(
                5 * min(M, N) * min(M, N) + 4 * min(M, N),
                2 * max(M, N) * min(M, N) + 2 * min(M, N) * min(M, N) + min(M, N),
            ),
            RWORK::Vector{$(real(dtype))} = Vector{$(real(dtype))}(undef, LRWORK),
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
                    Ptr{$dtype}, # A
                    Ref{Int64},   # LDA
                    Ptr{$(real(dtype))}, # S
                    Ptr{$dtype}, # U
                    Ref{Int64},   # LDU
                    Ptr{$dtype}, # VT
                    Ref{Int64},   # LDVT
                    Ptr{$dtype}, # WORK
                    Ref{Int64},   # LWORK
                    Ptr{$(real(dtype))}, # RWORK
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
end


function get_uvt(JOBU, JOBVT, M, N; dtype)
    if JOBU == Cchar('A')
        U = Matrix{dtype}(undef, M, M)
    elseif JOBU == Cchar('S')
        U = Matrix{dtype}(undef, M, min(M, N))
    else
        error("JOBU must be a Cchar with value 'A' or 'S'")
    end
    if JOBVT == Cchar('A')
        VT = Matrix{dtype}(undef, N, N)
    elseif JOBVT == Cchar('S')
        VT = Matrix{dtype}(undef, min(M, N), N)
    else
        error("JOBVT must be a Cchar with value 'A' or 'S'")
    end
    return U, VT
end

function get_uvt_view(JOBU, JOBVT, M, N, U, VT)
    if JOBU == Cchar('A')
        Uv = view(U, 1:M, 1:M)
    elseif JOBU == Cchar('S')
        Uv = view(U, 1:M, 1:min(M, N))
    else
        error("JOBU must be a Cchar with value 'A' or 'S'")
    end
    if JOBVT == Cchar('A')
        VTv = view(VT, 1:N, 1:N)
    elseif JOBVT == Cchar('S')
        VTv = view(VT, 1:min(M, N), 1:N)
    else
        error("JOBVT must be a Cchar with value 'A' or 'S'")
    end
    return Uv, VTv
end

include("./SVDFunctors.jl")



export svd_qr!,
    svd_divconquer!,
    svd_divconquer_cf64,
    svd_divconquer_cf32,
    svd_divconquer_f64,
    svd_divconquer_f32,
    svd_qr_cf32,
    svd_qr_cf64,
    svd_qr_f64,
    svd_qr_f32
