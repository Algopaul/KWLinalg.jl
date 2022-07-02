for (dtype, shorthand) in [(ComplexF64, "cf64"), (ComplexF32, "cf32")]
    name_dc = Symbol("svd_divconquer_$shorthand")
    name_qr = Symbol("svd_qr_$shorthand")
    @eval begin
        function $name_dc(
            M,
            N;
            JOBZ = Cchar('S'),
            LDA = max(1, M),
            S = Vector{$(real(dtype))}(undef, min(N, M)),
            UVT = get_uvt(JOBZ, JOBZ, M, N, dtype = $dtype),
            U = UVT[1],
            VT = UVT[2],
            LDU = max(1, stride(U, 2)),
            LDVT = max(1, stride(VT, 2)),
            LWORK = 2 * min(M, N) * min(M, N) + 2 * min(M, N) + max(M, N),
            WORK = Vector{$dtype}(undef, LWORK),
            LRWORK = max(
                5 * min(M, N) * min(M, N) + 4 * min(M, N),
                2 * max(M, N) * min(M, N) + 2 * min(M, N) * min(M, N) + min(M, N),
            ),
            RWORK = Vector{$(real(dtype))}(undef, LRWORK),
            IWORK = Vector{BlasInt}(undef, 8 * min(M, N)),
            INFO = 0,
        )
            return A -> svd_divconquer!(
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

        function $name_qr(
            M,
            N;
            JOBU = Cchar('S'),
            JOBVT = Cchar('S'),
            LDA = max(1, M),
            S = Vector{real($dtype)}(undef, min(M, N)),
            UVT = get_uvt(JOBU, JOBVT, M, N; dtype = $dtype),
            U = UVT[1],
            VT = UVT[2],
            LDU = max(1, stride(U, 2)),
            LDVT = max(1, stride(VT, 2)),
            LWORK = 10 * max(1, 2 * min(M, N) + max(M, N)),
            WORK = Vector{$dtype}(undef, LWORK),
            RWORK = Vector{real($dtype)}(undef, 5 * min(M, N)),
            INFO = 0,
        )
            return A -> svd_qr!(
                A;
                JOBU,
                JOBVT,
                M,
                N,
                LDA,
                S,
                U,
                LDU,
                VT,
                LDVT,
                LWORK,
                WORK,
                RWORK,
                INFO,
            )
        end

    end
end

for (dtype, shorthand) in [(Float64, "f64"), (Float32, "f32")]
    name_dc = Symbol("svd_divconquer_$shorthand")
    name_qr = Symbol("svd_qr_$shorthand")
    @eval begin
        function $name_dc(
            M,
            N;
            JOBZ = Cchar('S'),
            LDA = max(1, M),
            S = Vector{$(real(dtype))}(undef, min(N, M)),
            UVT = get_uvt(JOBZ, JOBZ, M, N, dtype = $dtype),
            U = UVT[1],
            VT = UVT[2],
            LDU = max(1, stride(U, 2)),
            LDVT = max(1, stride(VT, 2)),
            LWORK = get_lwork_gesdd_real(M, N, JOBZ),
            WORK = Vector{$dtype}(undef, LWORK),
            IWORK = Vector{BlasInt}(undef, 8 * min(M, N)),
            INFO = 0,
        )
            return A -> svd_divconquer!(
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
                IWORK,
                INFO,
            )
        end

        function $name_qr(
            M,
            N;
            JOBU = Cchar('S'),
            JOBVT = Cchar('S'),
            LDA = max(1, M),
            S = Vector{$dtype}(undef, min(M, N)),
            UVT = get_uvt(JOBU, JOBVT, M, N; dtype = $dtype),
            U = UVT[1],
            VT = UVT[2],
            LDU = max(1, stride(U, 2)),
            LDVT = max(1, stride(VT, 2)),
            LWORK = get_lwork_gesvd_real(M, N),
            WORK = Vector{$dtype}(undef, LWORK),
            INFO = 0,
        )
            return A -> svd_qr!(
                A;
                JOBU,
                JOBVT,
                M,
                N,
                LDA,
                S,
                U,
                LDU,
                VT,
                LDVT,
                LWORK,
                WORK,
                INFO,
            )
        end

    end
end

function svd_functor_divconquer(M, N, dtype; JOBZ = Cchar('S'))
    if dtype == Float64
        return svd_divconquer_f64(M, N; JOBZ)
    elseif dtype == Float32
        return svd_divconquer_f32(M, N; JOBZ)
    elseif dtype == ComplexF64
        return svd_divconquer_cf64(M, N; JOBZ)
    elseif dtype == ComplexF32
        return svd_divconquer_cf32(M, N; JOBZ)
    else
        error("unsupported dtype")
    end
end

function svd_functor_qr(M, N, dtype; JOBU = Cchar('S'), JOBVT = Cchar('S'))
    if dtype == Float64
        return svd_qr_f64(M, N; JOBU, JOBVT)
    elseif dtype == Float32
        return svd_qr_f32(M, N; JOBU, JOBVT)
    elseif dtype == ComplexF64
        return svd_qr_cf64(M, N; JOBU, JOBVT)
    elseif dtype == ComplexF32
        return svd_qr_cf32(M, N; JOBU, JOBVT)
    else
        error("unsupported dtype")
    end
end

export svd_functor_qr, svd_functor_divconquer
