# KWLinalg

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://algopaul.github.io/KWLinalg/)
[![Build status](https://github.com/Algopaul/KWLinalg/workflows/CI/badge.svg)](https://github.com/KWLinalg/actions?query=workflow%3ACI+branch%3Amain)
[![Coverage Status](http://codecov.io/github/Algopaul/KWLinalg/coverage.svg?branch=main)](http://codecov.io/github/Algopaul/KWLinalg?branch=main)

We provide wrappers for linear algebra routines that allow to pre-allocate memory for repeated executions of the same operations. For convenience, we also provide functors, that contain the necessary memory and can be called with no further allocations.

## Example

Running the code
```julia
using KWLinalg
using BenchmarkTools

m, n = 5, 3
dtype = Float64
A = rand(dtype, m, n)
AC = deepcopy(A)
f = svd_functor_divconquer(m, n, Float64)
function copy_and_svd_inplace!(A, AC, f)
    AC .= A
    f(AC)
    return nothing
end
@benchmark $copy_and_svd_inplace!($A, $AC, $f)
```
leads to following result:
```julia
BenchmarkTools.Trial: 10000 samples with 9 evaluations.
 Range (min … max):  2.612 μs …   8.526 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     2.659 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   2.672 μs ± 193.544 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

              ▄▇█▆▁                                            
  ▂▂▂▂▂▂▂▂▃▄▅██████▆▃▂▂▂▂▂▁▂▂▁▁▁▁▁▂▁▁▁▂▂▁▁▂▂▂▂▂▁▂▂▂▁▁▁▁▂▂▁▂▂▂ ▃
  2.61 μs         Histogram: frequency by time        2.81 μs <

 Memory estimate: 0 bytes, allocs estimate: 0.
```

In contrast, using the `LinearAlgebra` function `svd!` as in
```julia
using LinearAlgebra
using BenchmarkTools

m, n = 5, 3
dtype = Float64
A = rand(dtype, m, n)
AC = deepcopy(A)
function copy_and_svd!(A, AC)
    AC .= A
    svd!(AC)
    return nothing
end
@benchmark $copy_and_svd!($A, $AC)
```
leads to following result:
```julia
BenchmarkTools.Trial: 10000 samples with 8 evaluations.
 Range (min … max):  3.101 μs … 96.520 μs  ┊ GC (min … max): 0.00% … 94.23%
 Time  (median):     3.154 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   3.297 μs ±  1.473 μs  ┊ GC (mean ± σ):  0.75% ±  1.64%

  ▃▇█▇▅▃▁           ▁▁▁▁▁▂▃▂▂▂▂▃▃▂▁▂▂▂▁▁                     ▂
  ███████▄▄▄▁▅▁▅▇▇▅▇████████████████████████▇▆▅▆▅▅▆▅▄▅▆▆▆▆▆▆ █
  3.1 μs       Histogram: log(frequency) by time     4.14 μs <

 Memory estimate: 2.45 KiB, allocs estimate: 6.
```

The time-difference is mostly due to the overhead of garbage collection. `KWLinalg` provides a benefit, when linear algebra operations are performed on dense matrices of the same size for potentially millions of times (e.g. within an optimization loop).
 
 ## Installation
 
`KWLinalg` can be installed via `Pkg`:
 ```julia
 using Pkg
 Pkg.add(url="https://github.com/Algopaul/KWLinalg.git")
 ```
 
For a detailed description of the package and its functionality, we refer to the [documentation](https://algopaul.github.io/KWLinalg/).
