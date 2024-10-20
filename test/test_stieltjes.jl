using SingularIntegrals, ClassicalOrthogonalPolynomials, QuasiArrays, BandedMatrices, Test
using LazyBandedMatrices: blockcolsupport, Block, BlockHcat, blockbandwidths, paddeddata, colsupport, rowsupport
using ClassicalOrthogonalPolynomials: orthogonalityweight
using SingularIntegrals: Stieltjes, StieltjesPoint

@testset "Stieltjes" begin
    @testset "weights" begin
        w_T = ChebyshevTWeight()
        w_U = ChebyshevUWeight()
        w_P = LegendreWeight()
        x = axes(w_T,1)
        H = inv.(x .- x')
        @test iszero(H*w_T)
        @test (H*w_U)[0.1] ≈ π/10
        @test (H*w_P)[0.1] ≈ log(1.1) - log(1-0.1)

        @test H * w_T ≡ QuasiZeros{Float64}((x,))
        @test H * w_U == π*x
        @test (H * w_P)[0.1] ≈ log((0.1+1)/(1-0.1))

        w_T = orthogonalityweight(chebyshevt(0..1))
        w_U = orthogonalityweight(chebyshevu(0..1))
        w_P = orthogonalityweight(legendre(0..1))
        x = axes(w_T,1)
        H = inv.(x .- x')
        @test iszero(H*w_T)
        @test (H*w_U)[0.1] ≈ (2*0.1-1)*π
        @test (H*w_P)[0.1] ≈ (log(1+(-0.8)) - log(1-(-0.8)))
    end

    @testset "ops" begin
        wT = Weighted(ChebyshevT())
        wU = Weighted(ChebyshevU())
        x = axes(wT,1)
        H = inv.(x .- x')
        @test H isa Stieltjes{Float64,ChebyshevInterval{Float64}}

        @test (Ultraspherical(1) \ (H*wT))[1:10,1:10] == diagm(1 => fill(-π,9))
        @test (Chebyshev() \ (H*wU))[1:10,1:10] == diagm(-1 => fill(1.0π,9))

        # check consistency
        @test (Ultraspherical(1) \ (H*wT) * (wT \ wU))[1:10,1:10] ==
                    ((Ultraspherical(1) \ Chebyshev()) * (Chebyshev() \ (H*wU)))[1:10,1:10]
    end

    @testset "Other axes" begin
        x = Inclusion(0..1)
        y = 2x .- 1
        H = inv.(x .- x')
        T,U = ChebyshevT(),ChebyshevU()
        wT = Weighted(T)
        wU = Weighted(U)
        wT2 = wT[y,:]
        wU2 = wU[y,:]
        @test (Ultraspherical(1)[y,:]\(H*wT2))[1:10,1:10] == diagm(1 => fill(-π,9))
        @test (T[y,:]\(H*wU2))[1:10,1:10] == diagm(-1 => fill(1.0π,9))
    end

    @testset "Legendre" begin
        P = Legendre()
        x = axes(P,1)
        H = inv.(x .- x')
        Q = H*P
        @test Q[0.1,1:3] ≈ [log(0.1+1)-log(1-0.1), 0.1*(log(0.1+1)-log(1-0.1))-2,-3*0.1 + 1/2*(-1 + 3*0.1^2)*(log(0.1+1)-log(1-0.1))]
        X = jacobimatrix(P)
        @test Q[0.1,1:11]'*X[1:11,1:10] ≈ (0.1 * Array(Q[0.1,1:10])' - [2 zeros(1,9)])
    end

    @testset "mapped" begin
        T = chebyshevt(0..1)
        U = chebyshevu(0..1)
        x = axes(T,1)
        H = inv.(x .- x')
        @test_broken U\H*Weighted(T) isa BandedMatrix
    end
end

@testset "StieltjesPoint" begin
    T = Chebyshev()
    wT = Weighted(T)
    x = axes(wT,1)
    z = 0.1+0.2im
    S = inv.(z .- x')
    @test S isa StieltjesPoint{ComplexF64,ComplexF64,Float64,ChebyshevInterval{Float64}}

    @test S * ChebyshevWeight() ≈ π/(sqrt(z-1)sqrt(z+1))
    @test S * JacobiWeight(0.1,0.2) ≈ 0.051643014475741864 - 2.7066092318596726im

    f = wT * [[1,2,3]; zeros(∞)];
    J = T \ (x .* T)

    @test π*((z*I-J) \ f.args[2])[1,1] ≈ (S*f)[1]
    @test π*((z*I-J) \ f.args[2])[1,1] ≈ (S*f.args[1]*f.args[2])[1]

    x = Inclusion(0..1)
    y = 2x .- 1
    wT2 = wT[y,:]
    S = inv.(z .- x')
    f = wT2 * [[1,2,3]; zeros(∞)];

    @test (π/2*(((z-1/2)*I - J/2) \ f.args[2]))[1] ≈ (S*f.args[1]*f.args[2])[1]

    @testset "Real point" begin
        t = 2.0
        T,U,P = ChebyshevT(),ChebyshevU(),Legendre()
        x = axes(T,1)
        @test (inv.(t .- x') * Weighted(T))[1:10] ≈ (inv.((t+eps()im) .- x') * Weighted(T))[1:10]
        @test (inv.(t .- x') * Weighted(U))[1:10] ≈ (inv.((t+eps()im) .- x') * Weighted(U))[1:10]
        @test (inv.(t .- x') * P)[5] ≈ 0.0023221516632410816
        @test (inv.(t .- x') * P)[10] ≈ 2.2435707298304464E-6

        t = 2
        @test (inv.(t .- x') * Weighted(T))[1:10] ≈ (inv.((t+eps()im) .- x') * Weighted(T))[1:10]
        @test (inv.(t .- x') * Weighted(U))[1:10] ≈ (inv.((t+eps()im) .- x') * Weighted(U))[1:10]

        t = 0.5
        @test (inv.(t .- x') * Weighted(T))[1,1:3] ≈ [0,-π,-π]
        @test (inv.(t .- x') * Weighted(U))[1,1:3] ≈ [π/2,-π/2,-π]

        t = 0.5+0im
        @test (inv.(t .- x') * Weighted(T))[1,1:3] ≈ [0,-π,-π]
        @test (inv.(t .- x') * Weighted(U))[1,1:3] ≈ [π/2,-π/2,-π]
    end

    @testset "DimensionMismatch" begin
        x = Inclusion(0..1)
        z = 2.0
        @test_throws DimensionMismatch inv.(z .- x') * Weighted(ChebyshevT())
        @test_throws DimensionMismatch inv.(z .- x') * Weighted(ChebyshevU())
        @test_throws DimensionMismatch inv.(z .- x') * ChebyshevTWeight()
        @test_throws DimensionMismatch inv.(z .- x') * ChebyshevUWeight()
    end

    @testset "Matrix" begin
        z = [2.,3.]
        T = ChebyshevT()
        wT = Weighted(T)
        x = axes(wT,1)
        @test (inv.(z .- x') * wT)[:,1:100] ≈ ([(inv.(z[1] .- x') * wT)[1:100]'; (inv.(z[2] .- x') * wT)[1:100]'])
        f = wT * (T \ exp.(x))
        @test inv.(z .- x') * f ≈ [2.8826861116458593, 1.6307809018753612]
    end

    @testset "StieltjesPoints" begin
        z = [2.,3.]
        P = Legendre()
        x = axes(P,1)
        S = inv.(z .- x')
        @test (S * P)[:,1:5] ≈ [(inv.(2 .- x')*P)[1:5]'; (inv.(3 .- x')*P)[1:5]']

        @test S * JacobiWeight(0.1,0.2) == stieltjes.(Ref(JacobiWeight(0.1,0.2)), z)
    end
end


@testset "Ideal Fluid Flow" begin
    T = ChebyshevT()
    U = ChebyshevU()
    x = axes(U,1)
    H = inv.(x .- x')

    c = exp(0.5im)
    u = Weighted(U) * ((H * Weighted(U)) \ imag(c * x))

    ε  = eps();
    @test (inv.(0.1+ε*im .- x') * u + inv.(0.1-ε*im .- x') * u)/2 ≈ imag(c*0.1)
    @test real(inv.(0.1+ε*im .- x') * u ) ≈ imag(c*0.1)

    v = (s,t) -> (z = (s + im*t); imag(c*z) - real(inv.(z .- x') * u))
    @test v(0.1,0.2) ≈ 0.18496257285081724 # Emperical
end

@testset "OffStieltjes" begin
    @testset "ChebyshevU" begin
        U = ChebyshevU()
        W = Weighted(U)
        t = axes(U,1)
        x = Inclusion(2..3)
        T = chebyshevt(2..3)
        H = T \ inv.(x .- t') * W;

        @test MemoryLayout(H) isa PaddedLayout

        @test last(colsupport(H,1)) ≤ 20
        @test last(colsupport(H,6)) ≤ 40
        @test last(rowsupport(H)) ≤ 30
        @test T[2.3,1:100]'*(H * (W \ @.(sqrt(1-t^2)exp(t))))[1:100] ≈ 0.9068295340935111
        @test T[2.3,1:100]' * H[1:100,1:100] ≈ (inv.(2.3 .- t') * W)[:,1:100]

        u = (I + H) \ [1; zeros(∞)]
        @test u[3] ≈ -0.011220808241213699 #Emperical


        @testset "properties" begin
            U  = chebyshevu(T)
            X = jacobimatrix(U)
            Z = jacobimatrix(T)

            @test Z * H[:,1] - H[:,2]/2 ≈ [sum(W[:,1]); zeros(∞)]
            @test norm(-H[:,1]/2 + Z * H[:,2] - H[:,3]/2) ≤ 1E-12

            L = U \ ((x.^2 .- 1) .* Derivative(x) * T - x .* T)
            c = T \ sqrt.(x.^2 .- 1)
            @test [T[begin,:]'; L] \ [sqrt(2^2-1); zeros(∞)] ≈ c
        end
    end

    @testset "mapped" begin
        U = chebyshevu(-1..0)
        W = Weighted(U)
        t = axes(U,1)
        x = Inclusion(2..3)
        T = chebyshevt(2..3)
        H = T \ inv.(x .- t') * W
        N = 100
        @test T[2.3,1:N]' * H[1:N,1:N] ≈ (inv.(2.3 .- t') * W)[:,1:N]

        U = chebyshevu((-2)..(-1))
        W = Weighted(U)
        T = chebyshevt(0..2)
        x = axes(T,1)
        t = axes(W,1)
        H = T \ inv.(x .- t') * W
        @test T[0.5,1:N]'*(H * (W \ @.(sqrt(-1-t)*sqrt(t+2)*exp(t))))[1:N] ≈ 0.047390454610749054
    end

    @testset "expand" begin
        P = Legendre()
        @test stieltjes(P[:,1], 3) ≈ stieltjes(P,3)[1]
        @test stieltjes(P[:,1:3], 3) ≈ stieltjes(P,3)[:,1:3]
    end
end
