using SingularIntegrals, ClassicalOrthogonalPolynomials, ContinuumArrays, QuasiArrays, BandedMatrices, LinearAlgebra, Test
using SingularIntegrals: Stieltjes, StieltjesPoint, ChebyshevInterval, associated, Associated,
        orthogonalityweight, Weighted, *, dot, LogKernelPoint
using LazyArrays: MemoryLayout, PaddedLayout, colsupport, rowsupport, paddeddata
using LazyBandedMatrices: blockcolsupport, Block, BlockHcat, blockbandwidths

include("test_recurrence.jl")

@testset "Associated" begin
    T = ChebyshevT()
    U = ChebyshevU()
    @test associated(T) ≡ U
    @test associated(U) ≡ U
    @test Associated(T)[0.1,1:10] == Associated(U)[0.1,1:10] == U[0.1,1:10]

    P = Legendre()
    Q = Associated(P)
    x = axes(P,1)
    u = Q * (Q \ exp.(x))
    @test u[0.1] ≈ exp(0.1)
    @test grid(Q[:,Base.OneTo(5)]) ≈ eigvals(Matrix(jacobimatrix(Normalized(Q))[1:5,1:5]))

    w = orthogonalityweight(Q)
    @test axes(w,1) == axes(P,1)
    @test sum(w) == 1
end

include("test_hilbert.jl")
include("test_logkernel.jl")
include("test_power.jl")
include("test_piecewise.jl")