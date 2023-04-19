# SingularIntegrals.jl
A Julia package for computing singular integrals



[![Build Status](https://github.com/JuliaApproximation/SingularIntegrals.jl/workflows/CI/badge.svg)](https://github.com/JuliaApproximation/SingularIntegrals.jl/actions)
[![codecov](https://codecov.io/gh/JuliaApproximation/SingularIntegrals.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaApproximation/SingularIntegrals.jl)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaApproximation.github.io/SingularIntegrals.jl)


This package supports computing singular integrals involving Hilbert/Stieltjes/Cauchy,
log, and power law kernels.

Some examples:
```julia
julia> using SingularIntegrals, ClassicalOrthogonalPolynomials

julia> P = Legendre(); x = axes(P,1); f = expand(P, exp); #  expand exp(x) in Legendre polynomials

julia> @time inv.(10 .- x') * f # Stieltjes: ∫_{-1}^1 exp(x)/(10-x) dx
  0.000034 seconds (22 allocations: 2.266 KiB)
0.24332755428373515

julia> @time inv.(0.1+0im .- x') * f - inv.(0.1-0im .- x') * f ≈ -2π*im*exp(0.1) # example of Plemelj
  0.000052 seconds (49 allocations: 6.031 KiB)
true

julia> @time abs.(10 .- x') .^ 0.2 * f # Power law: ∫_{-1}^1 (10-x)^0.2 * exp(x) dx
  0.000077 seconds (21 allocations: 1.875 KiB)
3.7006631248289135

julia> @time abs.(0.3 .- x') .^ 0.2 * f # ∫_{-1}^1 abs(0.3-x)^0.2 * exp(x) dx
  0.000040 seconds (25 allocations: 2.172 KiB)
1.9044201526740234

julia> W = Weighted(ChebyshevU()); f = expand(W, x -> exp(x) * sqrt(1-x^2));

julia> @time log.(abs.(10 .- x')) * f # Log-kernel: ∫_{-1}^1 log(10-x) * exp(x) * sqrt(1-x^2) dx
  0.000040 seconds (14 allocations: 400 bytes)
4.043032838853287

julia> @time log.(abs.(0.3 .- x')) * f # ∫_{-1}^1 log(abs(0.3-x)) * exp(x) * sqrt(1-x^2) dx
  0.000035 seconds (116 allocations: 6.250 KiB)
-2.320391559008445
```
