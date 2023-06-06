using SingularIntegrals, ClassicalOrthogonalPolynomials, FillArrays, Test
import SingularIntegrals: RecurrenceArray

@testset "ComplexLogKernelPoint" begin

    P = Legendre()
    Pc = Legendre{ComplexF64}()
    x = axes(P,1)


    L = (z,k) -> sum(Pc / Pc \ (log.(z .- x).*P[:,k+1]))

    for z in (5, 1+2im, -1+2im, 1-2im, -3+0.0im, -3-0.0im)
        @test (log.(z .- x') * P)[1:5] ≈ L.(z, 0:4)
    end
    
    for z in (-5,-1,0,0.1)
        @test_throws DomainError log.(z .- x') * P
    end

end

@testset "LogKernelPoint" begin
    @testset "Complex point" begin
        wU = Weighted(ChebyshevU())
        x = axes(wU,1)
        z = 0.1+0.2im
        L = log.(abs.(z.-x'))
        @test L isa LogKernelPoint{Float64,ComplexF64,ComplexF64,Float64,ChebyshevInterval{Float64}}
    end

    @testset "Real point" begin
        U = ChebyshevU()
        x = axes(U,1)

        t = 2.0
        @test (log.(abs.(t .- x') )* Weighted(U))[1,1:3] ≈ [1.0362686329607178,-0.4108206734393296, -0.054364775221816465] #mathematica

        t = 0.5
        @test (log.(abs.(t .- x') )* Weighted(U))[1,1:3] ≈ [-1.4814921268505252, -1.308996938995747, 0.19634954084936207] #mathematica

        t = 0.5+0im
        @test (log.(abs.(t .- x') )* Weighted(U))[1,1:3] ≈ [-1.4814921268505252, -1.308996938995747, 0.19634954084936207] #mathematica
    end

    @testset "mapped" begin
        x = Inclusion(1..2)
        wU = Weighted(ChebyshevU())[affine(x, axes(ChebyshevU(),1)),:]
        x = axes(wU,1)
        z = 5
        L = log.(abs.(z .- x'))

        f = wU / wU \ @.(sqrt(2-x)sqrt(x-1)exp(x))
        @test L*f ≈ 2.2374312398976586 # MAthematica

        wU = Weighted(chebyshevu(1..2))
        f = wU / wU \ @.(sqrt(2-x)sqrt(x-1)exp(x))
        @test L*f ≈ 2.2374312398976586 # MAthematica
    end

    @testset "Legendre" begin
        z = 5
        P = Legendre()
        x = axes(P,1)

        L = (z,k) -> sum(P / P \ (log.(abs.(z .- x)).*P[:,k+1]))

        for z in (5, 1+2im, -1+2im, 1-2im, -3+0.0im, -3-0.0im, -3)
            @test (log.(abs.(z .- x')) * P)[1:5] ≈ L.(z, 0:4)
        end

        @testset "derivation" begin
            W = Weighted(Jacobi(1,1))
            
            @test L(0) ≈ 2log(z+1) + inv.(z .- x') * (P / P \ (x .- 1)) ≈ (1 + z)log(1 + z) - (z-1)log(z-1) - 2
            @test L(1) ≈ (inv.(z .- x') * W)[1]/(-2)

            @test z * L(0) ≈ 2z*log(z+1) + z*(inv.(z .- x') * (P / P \ (x .- 1))) ≈
                        2z*log(z+1) + (inv.(z .- x') * (P / P \ (x.^2 .- 1))) + (inv.(z .- x') * (P / P \ (1 .- x))) - 2 ≈
                        2L(1) - L(0) + 2(z+1)*log(z+1) - 2
            @test z * L(1) ≈ z * (inv.(z .- x') * W)[1]/(-2) ≈ -2/3 + (inv.(z .- x') * (x .* W))[1]/(-2) ≈
                        -2/3 + (inv.(z .- x') * W)[2]/(-4) ≈ -2/3 + L(2)

            for k = 2:5
                @test z * L(k) ≈ (k-1)/(2k+1)*L(k-1)+ (k+2)/(2k+1)*L(k+1)
                @test (2k+1)/(k+2)*z * L(k) ≈ (k-1)/(k+2)*L(k-1)+ L(k+1)
            end

            r0 = (1 + z)log(1 + z) - (z-1)log(z-1) - 2
            r1 = (z+1)*r0/2 + 1 - (z+1)log(z+1)
            r2 = z*r1 + 2*one(z)/3
            r = RecurrenceArray(z, ((1:2:∞)./(2:∞), Zeros(∞), (-1:∞)./(2:∞)), [r0,r1,r2])
            @test r[1:10] ≈ L.(0:9)
        end
    end
end


@testset "Log kernel" begin
    T = Chebyshev()
    wT = Weighted(Chebyshev())
    x = axes(wT,1)
    L = log.(abs.(x .- x'))
    D = T \ (L * wT)
    @test ((L * wT) * (T \ exp.(x)))[0.] ≈ -2.3347795490945797  # Mathematica

    x = Inclusion(-1..1)
    T = Chebyshev()[1x, :]
    L = log.(abs.(x .- x'))
    wT = Weighted(Chebyshev())[1x, :]
    @test (T \ (L*wT))[1:10,1:10] ≈ D[1:10,1:10]

    x = Inclusion(0..1)
    T = Chebyshev()[2x.-1, :]
    wT = Weighted(Chebyshev())[2x .- 1, :]
    L = log.(abs.(x .- x'))
    u =  wT * (2 *(T \ exp.(x)))
    @test u[0.1] ≈ exp(0.1)/sqrt(0.1-0.1^2)
    @test (L * u)[0.5] ≈ -7.471469928754152 # Mathematica

    @testset "mapped" begin
        T = chebyshevt(0..1)
        x = axes(T,1)
        L = log.(abs.(x .- x'))
        @test T[0.2,:]'*((T\L*Weighted(T)) * (T\exp.(x))) ≈ -2.9976362326874373 # Mathematica
    end
end