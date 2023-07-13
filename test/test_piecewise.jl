@testset "two-interval" begin
    T1,T2 = chebyshevt((-2)..(-1)), chebyshevt(0..2)
    U1,U2 = chebyshevu((-2)..(-1)), chebyshevu(0..2)
    W = PiecewiseInterlace(Weighted(U1), Weighted(U2))
    T = PiecewiseInterlace(T1, T2)
    U = PiecewiseInterlace(U1, U2)
    x = axes(W,1)
    H = T \ inv.(x .- x') * W;

    @test iszero(H[1,1])
    @test H[3,1] ≈ π
    @test maximum(blockcolsupport(H,Block(5))) ≤ Block(50)
    @test blockbandwidths(H) == (25,26)

    c = W \ broadcast(x -> exp(x)* (0 ≤ x ≤ 2 ? sqrt(2-x)*sqrt(x) : sqrt(-1-x)*sqrt(x+2)), x)
    f = W * c
    @test T[0.5,1:200]'*(H*c)[1:200] ≈ -6.064426633490422

    @testset "inversion" begin
        H̃ = BlockHcat(Eye((axes(H,1),))[:,Block(1)], H)
        @test blockcolsupport(H̃,Block(1)) == Block.(1:1)
        @test last(blockcolsupport(H̃,Block(2))) ≤ Block(30)

        UT = U \ T
        D = U \ Derivative(x) * T
        V = x -> x^4 - 10x^2
        Vp = x -> 4x^3 - 20x
        V_cfs = T \ V.(x)
        Vp_cfs_U = D * V_cfs
        Vp_cfs_T = T \ Vp.(x);

        @test (UT \ Vp_cfs_U)[Block.(1:10)] ≈ Vp_cfs_T[Block.(1:10)]

        @time c = H̃ \ Vp_cfs_T;

        @test c[Block.(1:100)] ≈ H̃[Block.(1:100),Block.(1:100)] \ Vp_cfs_T[Block.(1:100)]

        E1,E2 = c[Block(1)]
        @test [E1,E2] ≈  [12.939686758642496,-10.360345667126758]
        c1 = [paddeddata(c)[3:2:end]; Zeros(∞)]
        c2 = [paddeddata(c)[4:2:end]; Zeros(∞)]

        u1 = Weighted(U1) * c1
        u2 = Weighted(U2) * c2
        x1 = axes(u1,1)
        x2 = axes(u2,1)

        @test inv.(-1.3 .- x1') * u1 + inv.(-1.3 .- x2') * u2 + E1 ≈ Vp(-1.3)
        @test inv.(1.3 .- x1') * u1 + inv.(1.3 .- x2') * u2 + E2 ≈ Vp(1.3)
    end

    @testset "Stieltjes" begin
        z = 5.0
        @test inv.(z .- x')*f ≈ 1.317290060427562

        t = 1.2
        @test inv.(t .- x')*f ≈ -2.797995066227555
        @test log.(abs.(t .- x'))*f ≈ -5.9907385495482821485
        @test log.(abs.(z .- x'))*f ≈ 6.523123127595374
        @test log.(abs.((-z) .- x'))*f ≈ 8.93744698863906
    end
end

@testset "three-interval" begin
    d = (-2..(-1), 0..1, 2..3)
    T = PiecewiseInterlace(chebyshevt.(d)...)
    U = PiecewiseInterlace(chebyshevu.(d)...)
    W = PiecewiseInterlace(Weighted.(U.args)...)
    x = axes(W,1)
    H = T \ inv.(x .- x') * W
    c = W \ broadcast(x -> exp(x) *
        if -2 ≤ x ≤ -1
            sqrt(x+2)sqrt(-1-x)
        elseif 0 ≤ x ≤ 1
            sqrt(1-x)sqrt(x)
        else
            sqrt(x-2)sqrt(3-x)
        end, x)
    f = W * c
    @test T[0.5,1:200]'*(H*c)[1:200] ≈ -3.0366466972156143
end