module KWLinalg

using LinearAlgebra
import LinearAlgebra:
    require_one_based_indexing,
    chkstride1,
    BlasInt,
    LAPACK.chkargsok,
    LAPACK.getrf!,
    LinearAlgebra.BLAS.libblastrampoline

for (getrf, dtype) in [(:dgetrf_, Float64), (:zgetrf_, ComplexF64)]
    @eval begin
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
        end
    end
end

end
