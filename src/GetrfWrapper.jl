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
