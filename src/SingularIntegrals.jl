module SingularIntegrals
using ClassicalOrthogonalPolynomials, ContinuumArrays, QuasiArrays, LazyArrays, LinearAlgebra
using ContinuumArrays: @simplify, Weight, AbstractAffineQuasiVector
using QuasiArrays: AbstractQuasiMatrix, BroadcastQuasiMatrix, LazyQuasiArrayStyle
using ClassicalOrthogonalPolynomials: AbstractJacobiWeight, WeightedBasis
using LazyArrays: AbstractCachedMatrix
import Base: *, +, -, /, \, Slice

include("stieltjes.jl")
include("power.jl")


### generic fallback
for Op in (:Hilbert, :StieltjesPoint, :LogKernelPoint, :PowKernelPoint, :LogKernel, :PowKernel)
    @eval begin
        @simplify function *(H::$Op, wP::WeightedBasis{<:Any,<:Weight,<:Any})
            w,P = wP.args
            Q = OrthogonalPolynomial(w)
            (H * Weighted(Q)) * (Q \ P)
        end
        @simplify *(H::$Op, wP::Weighted{<:Any,<:SubQuasiArray{<:Any,2}}) = H * view(Weighted(parent(wP.P)), parentindices(wP.P)...)
    end
end


end # module SingularIntegrals
