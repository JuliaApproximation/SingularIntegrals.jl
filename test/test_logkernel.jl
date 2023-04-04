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