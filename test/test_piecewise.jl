using ClassicalOrthogonalPolynomials, SingularIntegrals, StaticArrays, Test
using ClassicalOrthogonalPolynomials: UnionDomain
using ContinuumArrays: ⊎, PiecewiseBasis
using LazyBandedMatrices: blocklengths

@testset "two-interval" begin
    T1,T2 = chebyshevt((-2)..(-1)), chebyshevt(0..2)
    U1,U2 = chebyshevu((-2)..(-1)), chebyshevu(0..2)
    W = PiecewiseInterlace(Weighted(U1), Weighted(U2))
    T = PiecewiseInterlace(T1, T2)
    U = PiecewiseInterlace(U1, U2)
    x = axes(W,1)
    H = T \ pinv.(x .- x') * W

    @test iszero(H[1,1])
    @test H[3,1] ≈ π
    @test maximum(blockcolsupport(H,Block(5))) ≤ Block(50)
    @test blockbandwidths(H) == (25,26)

    c = W \ broadcast(x -> exp(x)* (0 ≤ x ≤ 2 ? sqrt(2-x)*sqrt(x) : sqrt(-1-x)*sqrt(x+2)), x)
    f = W * c
    @test T[0.5,1:200]'*(H*c)[1:200] ≈ -6.064426633490422

    @testset "inversion" begin
        H̃ = BlockHcat(Eye((axes(H,1),))[:,Block(1)], H)
        @test blockcolsupport(H̃,Block(1)) == Block.(1:1)
        @test last(blockcolsupport(H̃,Block(2))) ≤ Block(40)

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

        @test pinv.(-1.3 .- x1') * u1 + inv.(-1.3 .- x2') * u2 + E1 ≈ Vp(-1.3)
        @test inv.(1.3 .- x1') * u1 + pinv.(1.3 .- x2') * u2 + E2 ≈ Vp(1.3)
    end

    @testset "Stieltjes" begin
        z = 5.0
        @test inv.(z .- x')*f ≈ 1.317290060427562

        t = 1.2
        @test pinv.(t .- x')*f ≈ -2.797995066227555
        @test log.(abs.(t .- x'))*f ≈ -5.9907385495482821485
        @test log.(abs.(z .- x'))*f ≈ 6.523123127595374
        @test log.(abs.((-z) .- x'))*f ≈ 8.93744698863906

        zs = [5.0, 3.0+im]
        Sz = stieltjes(W, zs)
        for k in eachindex(zs)
            @test Sz[k,1:10] ≈ stieltjes(W, zs[k])[1:10]
        end
    end
end

@testset "three-interval" begin
    d = (-2..(-1), 0..1, 2..3)
    T = PiecewiseInterlace(chebyshevt.(d)...)
    U = PiecewiseInterlace(chebyshevu.(d)...)
    W = PiecewiseInterlace(Weighted.(U.args)...)
    x = axes(W,1)
    H = T \ pinv.(x .- x') * W
    c = W \ broadcast(x -> exp(x) *
        if -2 ≤ x ≤ -1
            sqrt(x+2)sqrt(-1-x)
        elseif 0 ≤ x ≤ 1
            sqrt(1-x)sqrt(x)
        else
            sqrt(x-2)sqrt(3-x)
        end, x)
    f = W * c
    @test T[0.5,1:200]'*(H*c)[1:200] ≈ -3.0366466972156143
end

@testset "piecewise at a point" begin
    d3 = (-1..0, 0..1, 2..3)
    h = expand(exp(x) for x in UnionDomain(d3...))
    hs = [expand(exp(x) for x in c) for c in d3]
    @test stieltjes(h, 5.0) ≈ sum(stieltjes.(hs, 5.0))
    @test hilbert(h, 2.5) ≈ hilbert(hs[3], 2.5) + (stieltjes(hs[1], 2.5) + stieltjes(hs[2], 2.5))/π

    @testset "vector-valued" begin
        g = x -> [exp(-40(x-0.1)^2); cos(x-0.1)*exp(-40(x-0.1)^2)]
        𝐟 = expand(g(x) for x in UnionDomain(-1..0, 0..1))
        𝐟₁ = expand(g(x) for x in -1..0)
        𝐟₂ = expand(g(x) for x in 0..1)
        @test stieltjes(𝐟, im) isa Vector{ComplexF64}
        for z in (im, 2.0, 0.3+0.1im)
            @test stieltjes(𝐟, z) ≈ stieltjes(𝐟₁, z) + stieltjes(𝐟₂, z)
            @test cauchy(𝐟, z) ≈ cauchy(𝐟₁, z) + cauchy(𝐟₂, z)
        end
        @test hilbert(𝐟, 0.3) ≈ hilbert(𝐟₂, 0.3) + stieltjes(𝐟₁, 0.3)/π
        @test hilbert(𝐟, -0.5) ≈ hilbert(𝐟₁, -0.5) + stieltjes(𝐟₂, -0.5)/π

        𝐬 = expand(SVector(exp(x), cos(x)) for x in UnionDomain(d3...))
        @test stieltjes(𝐬, im) isa SVector{2,ComplexF64}
        @test stieltjes(𝐬, im) ≈ SVector(stieltjes(h, im), sum(stieltjes.([expand(cos(x) for x in c) for c in d3], im)))
    end
end

@testset "PiecewiseBasis" begin
    n = 5
    f₁ = legendre(0..1)[:,1:n] * [1, -2, 3, 0.5, 1]
    f₂ = legendre(-1..0)[:,1:n] * [2, 1, -1, 0.25, 3]
    f = f₁ ⊎ f₂
    P = basis(f)
    @test P isa PiecewiseBasis
    @test collect(blocklengths(axes(stieltjes(P, im), 2))) == collect(blocklengths(axes(P, 2))) == [n, n]

    for z in (im, 2.0, 0.3+0.1im)
        @test stieltjes(f, z) ≈ stieltjes(f₁, z) + stieltjes(f₂, z)
        @test cauchy(f, z) ≈ cauchy(f₁, z) + cauchy(f₂, z)
    end
    zs = [im, 2.0, 0.3+0.1im]
    @test stieltjes(f, zs) ≈ stieltjes(f₁, zs) + stieltjes(f₂, zs)
    @test hilbert(f, 0.3) ≈ hilbert(f₁, 0.3) + stieltjes(f₂, 0.3)/π
    @test hilbert(f, -0.5) ≈ hilbert(f₂, -0.5) + stieltjes(f₁, -0.5)/π

    x = axes(f,1)
    @test inv.(im .- x') * f ≈ stieltjes(f, im)
end
