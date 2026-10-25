# SingularIntegrals.jl

A Julia package for computing singular integrals involving Hilbert/Stieltjes/Cauchy,
log, and power law kernels.

## Examples

Expand ``\exp(x)`` in Legendre polynomials and compute singular integrals against it:

```@repl index
using SingularIntegrals, ClassicalOrthogonalPolynomials
P = Legendre(); x = axes(P, 1); f = expand(P, exp);
inv.(10 .- x') * f # Stieltjes: ∫₋₁¹ exp(x) / (10-x) dx
inv.(0.1+0im .- x') * f - inv.(0.1-0im .- x') * f ≈ -2π*im*exp(0.1) # example of Plemelj
```

Power law kernels are supported for Legendre and weighted ultraspherical bases:

```@repl index
(abs.(10 .- x') .^ 0.2 * Legendre())[1:5] # ∫₋₁¹ (10-x)^0.2 * P_k(x) dx for k = 0,…,4
```

Log kernels are supported for weighted bases:

```@repl index
W = Weighted(ChebyshevU()); f = expand(W, x -> exp(x) * sqrt(1-x^2));
log.(abs.(10 .- x')) * f # Log-kernel: ∫₋₁¹ log(10-x) * exp(x) * sqrt(1-x^2) dx
log.(abs.(0.3 .- x')) * f # ∫₋₁¹ log(abs(0.3-x)) * exp(x) * sqrt(1-x^2) dx
```

## API

```@autodocs
Modules = [SingularIntegrals]
```
