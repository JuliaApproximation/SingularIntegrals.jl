module SingularIntegrals
using ClassicalOrthogonalPolynomials, ContinuumArrays, QuasiArrays, LazyArrays, LazyBandedMatrices, FillArrays, BandedMatrices, LinearAlgebra, SpecialFunctions, HypergeometricFunctions, InfiniteArrays
using ContinuumArrays: @simplify, Weight, AbstractAffineQuasiVector, inbounds_getindex, broadcastbasis
using QuasiArrays: AbstractQuasiMatrix, BroadcastQuasiMatrix, LazyQuasiArrayStyle
import ClassicalOrthogonalPolynomials: AbstractJacobiWeight, WeightedBasis, jacobimatrix, orthogonalityweight, recurrencecoefficients, _p0, Clenshaw, chop, initiateforwardrecurrence
using LazyBandedMatrices: Tridiagonal, SymTridiagonal, subdiagonaldata, supdiagonaldata, diagonaldata
import LazyArrays: AbstractCachedMatrix, AbstractCachedArray, paddeddata, arguments, resizedata!, cache_filldata!, zero!, cacheddata
import Base: *, +, -, /, \, Slice, axes, getindex, sum, ==, oneto, size, broadcasted, copy, tail
import LinearAlgebra: dot
using BandedMatrices: _BandedMatrix
using FastTransforms: _forwardrecurrence!, _forwardrecurrence_next

export associated


include("recurrence.jl")
include("stieltjes.jl")
include("logkernel.jl")
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
