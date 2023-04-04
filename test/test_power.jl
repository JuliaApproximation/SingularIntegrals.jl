using SingularIntegrals, ClassicalOrthogonalPolynomials, Test
using SingularIntegrals: PowKernelPoint

#################################################
# ∫f(x)g(x)(t-x)^a dx evaluation where f and g in Legendre
#################################################

@testset "Pow kernel" begin
    @testset "Multiplication methods" begin
        P = Normalized(Legendre())
        x = axes(P,1)
        for (a,t) in ((0.1,1.2), (0.5,1.5))
            @test (t.-x).^a isa PowKernelPoint
            w = (t.-x).^a
            @test w .* P isa typeof(P*PowerLawMatrix(P,a,t))
        end
        # some functions
        f = P \ exp.(x.^2)
        g = P \ (sin.(x).*exp.(x.^(2)))
        # some parameters for (t-x)^a
        a = BigFloat("1.23")
        t = BigFloat("1.00001")
        # define powerlaw multiplication
        w = (t.-x).^a

        # check if it can compute the integral correctly
        @test g'*(P'*(w.*P)*f) ≈ -2.656108697646584 # Mathematica
    end
    @testset "Equivalence to multiplication in integer case" begin
        # TODO: overload integer input to make this work
        P = Normalized(Legendre())
        x = axes(P,1)
        a = 1
        t = 1.2
        @test_broken PowerLawMatrix(P,Float64(a),t)[1:20,1:20] ≈ ((t*I-jacobimatrix(P))^a)[1:20,1:20]
        a = 2
        t = 1.0001
        J = ((t*I-jacobimatrix(P)))[1:80,1:80]
        @test_broken PowerLawMatrix(P,BigFloat("$a"),BigFloat("$t"))[1:60,1:60] ≈ (J^2)[1:60,1:60]
    end
    @testset "Cached Legendre power law integral operator" begin
        P = Normalized(Legendre())
        a = 2*rand(1)[1]
        t = 1.0000000001
        Acached = PowerLawMatrix(P,BigFloat("$a"),BigFloat("$t"))
        @test size(Acached) == (∞,∞)
    end
    @testset "PowKernelPoint dot evaluation" begin
        @testset "Set 1" begin
                P = Normalized(Legendre())
                x = axes(P,1)
                f = P \ abs.(π*x.^7)
                g = P \ (cosh.(x.^3).*exp.(x.^(2)))
                a = 1.9127
                t = 1.211
                w = (BigFloat("$t") .- x).^BigFloat("$a")
                Pw = P'*(w .* P)
                @test w isa PowKernelPoint
                @test Pw[1:20,1:20] ≈ PowerLawMatrix(P,a,t)[1:20,1:20]
                # this is slower than directly using PowerLawMatrix but it works
                @test dot(f[1:20],Pw[1:20,1:20],g[1:20]) ≈ 5.082145576355614 # Mathematica
            end
        @testset "Set 2" begin
            P = Normalized(Legendre())
            x = axes(P,1)
            f = P \ exp.(x.^2)
            g = P \ (sin.(x).*exp.(x.^(2)))
            a = 1.23
            t = 1.00001
            W = PowerLawMatrix(P,a,t)
            @test dot(f,W,g) ≈ -2.656108697646584 # Mathematica
        end
        @testset "Set 3" begin
            P = Normalized(Legendre())
            x = axes(P,1)
            t = 1.2
            a = 1.1
            W = PowerLawMatrix(P,a,t)
            f = P \ exp.(x)
            g = P \ exp.(x.^2)
            @test dot(f,W,g) ≈ 2.916955525390389 # Mathematica
        end
        @testset "Set 4" begin
            P = Normalized(Legendre())
            x = axes(P,1)
            t = 1.001
            a = 1.001
            W = PowerLawMatrix(P,a,t)
            f = P \ (sinh.(x).*exp.(x))
            g = P \ cos.(x.^3)
            @test dot(f,W,g) ≈ -0.1249375144525209 # Mathematica
        end
        @testset "More explicit evaluation tests" begin
            # basis
            a = 2.9184
            t = 1.000001
            P = Normalized(Legendre())
            x = axes(P,1)
            # operator
            W = PowerLawMatrix(P,a,t)
            # functions
            f = P \ exp.(x)
            g = P \ sin.(x)
            const1(x) = 1
            onevec = P \ const1.(x)
            # dot() and * methods tests, explicit values via Mathematica
            @test -2.062500116206712 ≈ dot(onevec,W,g)
            @test 2.266485452423447 ≈ dot(onevec,W,f)
            @test -0.954305839543464 ≈ dot(g,W,f)
            @test 1.544769699288028 ≈ dot(f,W,f)
            @test 1.420460011606107 ≈ dot(g,W,g)
        end
    end
    @testset "Tests for -1 < a < 0" begin
        P = Normalized(Legendre())
        x = axes(P,1)
        a = -0.7
        t = 1.271
        # operator
        W = PowerLawMatrix(P,a,t)
        WB = PowerLawMatrix(P,BigFloat("$a"),BigFloat("$t"))
        # functions
        f0 = P \ exp.(2 .*x.^2)
        g0 = P \ sin.(x)
        @test dot(f0,W,g0) ≈  dot(f0,WB,g0) ≈ 1.670106472636101 # Mathematica
        f1 = P \ ((x.^2)./3 .+(x.^3)./3)
        g1 = P \ (x.*exp.(x.^3))
        @test dot(f1,W,g1) ≈ dot(f1,WB,g1) ≈ 0.5362428541997497 # Mathematica
    end
    @testset "Lanczos" begin
        P = Normalized(Legendre())
        x = axes(P,1)
        @time D = ClassicalOrthogonalPolynomials.LanczosData((1.001 .- x).^0.5, P);
        @time ClassicalOrthogonalPolynomials.resizedata!(D,100);
    end

    @testset "Jacobi" begin
        P = Weighted(Jacobi(0.1,0.2))
        x = axes(P,1)
        S = abs.(x .- x').^0.5
        @test S isa PowKernel
    end
end

using HypergeometricFunctions


α,λ = 0.2,0.4
T = Float64

x = 10.0

sqrt(convert(T,π))gamma(λ+one(T)/2)abs(x)^α*_₂F₁((1-α)/2, -α/2, 1+λ, 1/x^2)/gamma(1+λ)



