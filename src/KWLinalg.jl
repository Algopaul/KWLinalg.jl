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

include("./GetrfWrapper.jl")
include("./SVDWrapper.jl")

end

