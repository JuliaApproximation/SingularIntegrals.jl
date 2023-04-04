using SingularIntegrals, ClassicalOrthogonalPolynomials, Test
using SingularIntegrals: PowerKernelPoint


# using HypergeometricFunctions


# α,λ = 0.2,0.4
# T = Float64

# x = 10.0


@testset "Weights" begin
    z = 10.0
    x = axes(ChebyshevT(),1)
    L = abs.(z .- x') .^ 0.1
    # from Mathmatica
    @test L* UltrasphericalWeight(1) ≈ 1.9772924292721128
    @test L* LegendreWeight() ≈ 2.517472100701719
end
