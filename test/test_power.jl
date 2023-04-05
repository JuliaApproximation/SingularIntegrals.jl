using SingularIntegrals, ClassicalOrthogonalPolynomials, Test
using SingularIntegrals: PowerKernelPoint, powerlawmoment, powerlawrecurrence, RecurrenceArray


# using HypergeometricFunctions


# α,λ = 0.2,0.4
# T = Float64

# x = 10.0


@testset "Weights" begin
    @testset "on interval" begin
        z = 0.2
        x = axes(ChebyshevT(),1)
        α = 0.1
        L = abs.(z .- x') .^ α
        λ = 1
        L0 = powerlawmoment(Val(0), α, λ, z)
        # L1 = powerlawmoment(Val(1), α, λ, z)

        @test L* UltrasphericalWeight(λ) ≈ L0  ≈ 1.407056672015012 # from Mathmatica
    end
    @testset "off interval" begin
        z = 10.0
        x = axes(ChebyshevT(),1)
        α = 0.1
        L = abs.(z .- x') .^ α
        λ = 1
        L0 = powerlawmoment(Val(0), α, λ, z)
        L1 = powerlawmoment(Val(1), α, λ, z)
        @test L* UltrasphericalWeight(λ) ≈ L0  ≈ 1.9772924292721128 # from Mathmatica
        @test L1 ≈ -0.009901716900034385
        A, B, C = powerlawrecurrence(α, λ)
        @test (A[2]z + B[2])*L1-C[2]L0 ≈ -0.00022324029766696007
        r = RecurrenceArray(z, (A,B,C), [L0,L1])
        @test r[5] ≈ -2.5742591209035326E-7 
        @test r[1:10] ≈ (L * Weighted(Ultraspherical(λ)))[1,1:10]

        @test L* LegendreWeight() ≈ 2.517472100701719
        @test (L * Legendre())[5] ≈ -1.3328397976790363E-7 
    end
end
