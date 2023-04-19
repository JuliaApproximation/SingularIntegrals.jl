using SingularIntegrals, ClassicalOrthogonalPolynomials, Test
using SingularIntegrals: PowerKernelPoint, powerlawmoment, powerlawrecurrence, RecurrenceArray

@testset "Weights" begin
    z = 10.0
    x = axes(ChebyshevT(),1)
    α = 0.1
    L = abs.(z .- x') .^ α
    # from Mathmatica
    λ = 1
    L0 = powerlawmoment(Val(0), α, λ, z)
    L1 = powerlawmoment(Val(1), α, λ, z)
    @test L* UltrasphericalWeight(λ) ≈ L0  ≈ 1.9772924292721128
    @test L1 ≈ -0.009901716900034385
    A, B, C = powerlawrecurrence(α, λ)
    @test (A[2]z + B[2])*L1-C[2]L0 ≈ -0.00022324029766696007
    r = RecurrenceArray(z, (A,B,C), [L0,L1])
    @test r[5] ≈ -2.5742591209035326E-7 
    @test r[1:10] ≈ (L * Weighted(Ultraspherical(λ)))[1,1:10]

    @test L* LegendreWeight() ≈ 2.517472100701719
    @test (L * Legendre())[5] ≈ -1.3328397976790363E-7 
end

@testset "Compare with Mathematica results" begin
    @testset "z >> 1" begin
        z = 10.0
        x = axes(ChebyshevT(),1)
        α = 0.1
        L = abs.(z .- x') .^ α
        # from Mathematica
        λ = 1
        L0 = powerlawmoment(Val(0), α, λ, z)
        L1 = powerlawmoment(Val(1), α, λ, z)
        @test L* UltrasphericalWeight(λ) ≈ L0  ≈ 1.9772924292721128
        @test L1 ≈ -0.009901716900034385
        A, B, C = powerlawrecurrence(α, λ)
        @test (A[2]z + B[2])*L1-C[2]L0 ≈ -0.00022324029766696007
        r = RecurrenceArray(z, (A,B,C), [L0,L1])
        @test r[5] ≈ -2.5742591209035326E-7 
        @test r[1:10] ≈ (L * Weighted(Ultraspherical(λ)))[1,1:10]

        @test L* LegendreWeight() ≈ 2.517472100701719
        @test (L * Legendre())[5] ≈ -1.3328397976790363E-7 
    end
    @testset "z > 1" begin
        z = 1.1
        x = axes(ChebyshevT(),1)
        α = 0.1
        L = abs.(z .- x') .^ α
        # from Mathematica
        λ = 1
        L0 = powerlawmoment(Val(0), α, λ, z)
        L1 = powerlawmoment(Val(1), α, λ, z)
        @test L* UltrasphericalWeight(λ) ≈ L0  ≈ 1.56655191643602910
        @test L1 ≈ -0.0850992609853987
        A, B, C = powerlawrecurrence(α, λ)
        r = RecurrenceArray(z, (A,B,C), [L0,L1])
        @test r[5] ≈ -0.00381340800899034
    end
    @testset "z < -1" begin
        z = -1.43
        x = axes(ChebyshevT(),1)
        α = 0.1
        L = abs.(z .- x') .^ α
        # from Mathematica
        λ = 1
        L0 = powerlawmoment(Val(0), α, λ, z)
        L1 = powerlawmoment(Val(1), α, λ, z)
        @test L* UltrasphericalWeight(λ) ≈ L0  ≈ 1.617769472203235
        @test L1 ≈ 0.06180254575531147
        A, B, C = powerlawrecurrence(α, λ)
        r = RecurrenceArray(z, (A,B,C), [L0,L1])
        @test r[5] ≈ -0.000807806617344142
    end
    @testset "1 < z < 1" begin
        z = 0.7398
        x = axes(ChebyshevT(),1)
        α = 0.1
        L = abs.(z .- x') .^ α
        # from Mathematica
        λ = 1
        L0 = powerlawmoment(Val(0), α, λ, z)
        L1 = powerlawmoment(Val(1), α, λ, z)
        @test L* UltrasphericalWeight(λ) ≈ L0  ≈ 1.480874346601822
        @test L1 ≈ -0.1328257513137366
        A, B, C = powerlawrecurrence(α, λ)
        r = RecurrenceArray(z, (A,B,C), [L0,L1])
        @test r[5] ≈ 0.02537001925265781
    end
end