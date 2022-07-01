# KWLinalg

Collection of functions to perform linear algebra operations in-place.

## Non allocating updates LU decompositions

We extend the function `LAPACK.getrf!` to provide a pivot-vector `ipiv` as second argument. In this way, `ipiv` can be preallocated and `LAPACK.getrf!(A, ipiv)` does not cause any allocations. This allows one to update LU decompositions in-place.

For a matrix `A` the function `lu!(A)` still allocates workspaces.
```julia
using LinearAlgebra
n = 5
A = rand(n, n)
@allocated D = lu(A) # 384 bytes
@allocated Di = lu!(A) # 128 bytes
```

The function `LAPACK.getrf!` is only available to matrices and always allocates the pivot-vector `ipiv`. We provide a function `LAPACK.getrf!(A, ipiv)`, such that a pivot vector can be provied and is not allocated everytime. For an lu-decomposition object, this can be used as follows
```julia
using KWLinalg
using LinearAlgebra
n = 5
A = rand(n, n)
D = lu(A)
# A is updated
A .= rand(n, n)
# We update the LU-decomposition in-place.
LAPACK.getrf!(D.factors, D.ipiv)
```

```@docs
KWLinalg.getrf!
```

## Non allocating singular value decompositions

