module SingularIntegrals
using ClassicalOrthogonalPolynomials, ContinuumArrays, QuasiArrays, LazyArrays, LazyBandedMatrices, FillArrays, BandedMatrices, LinearAlgebra, SpecialFunctions, HypergeometricFunctions, InfiniteArrays
using ContinuumArrays: @simplify, Weight, AbstractAffineQuasiVector, inbounds_getindex, broadcastbasis, MappedBasisLayouts, MemoryLayout, MappedWeightLayout, AbstractWeightLayout, ExpansionLayout, demap, basismap, AbstractBasisLayout, SubBasisLayout
using QuasiArrays: AbstractQuasiMatrix, BroadcastQuasiMatrix, LazyQuasiArrayStyle, AbstractQuasiVecOrMat
import ClassicalOrthogonalPolynomials: AbstractJacobiWeight, WeightedBasis, jacobimatrix, orthogonalityweight, recurrencecoefficients, _p0, Clenshaw, chop, initiateforwardrecurrence, MappedOPLayouts, unweighted
using LazyBandedMatrices: Tridiagonal, SymTridiagonal, subdiagonaldata, supdiagonaldata, diagonaldata, ApplyLayout
import LazyArrays: AbstractCachedMatrix, AbstractCachedArray, paddeddata, arguments, resizedata!, cache_filldata!, zero!, cacheddata
import Base: *, +, -, /, \, Slice, axes, getindex, sum, ==, oneto, size, broadcasted, copy, tail, view
import LinearAlgebra: dot
using BandedMatrices: _BandedMatrix
using FastTransforms: _forwardrecurrence!, _forwardrecurrence_next

export associated, stieltjes, logkernel, powerkernel, complexlogkernel


include("recurrence.jl")
include("stieltjes.jl")
include("logkernel.jl")
include("power.jl")


### generic fallback
for Op in (:Stieltjes, :StieltjesPoint, :LogKernelPoint, :PowerKernelPoint, :LogKernel)
    @eval begin
        @simplify function *(H::$Op, wP::WeightedBasis{<:Any,<:Weight,<:Any})
            w,P = wP.args
            Q = OrthogonalPolynomial(w)
            (H * Weighted(Q)) * (Q \ P)
        end
        @simplify *(H::$Op, wP::Weighted{<:Any,<:SubQuasiArray{<:Any,2}}) = H * view(Weighted(parent(wP.P)), parentindices(wP.P)...)
    end
end



# general routines
for lk in (:logkernel, :complexlogkernel, :stieltjes)
    lk_layout = Symbol(lk, :_layout)
    @eval begin
        $lk_layout(::AbstractBasisLayout, P, z...) = error("not implemented")
        $lk_layout(::AbstractWeightLayout, w, zs::AbstractVector) = [stieltjes(w, z) for z in zs]
        function $lk_layout(::AbstractWeightLayout, w, z::Inclusion)
            axes(w,1) == z || error("Not implemented")
            $lk(w)
        end
        function $lk_layout(::AbstractBasisLayout, w, z::Inclusion)
            axes(w,1) == z || error("Not implemented")
            $lk(w)
        end
        $lk_layout(lay, P, z...) = $lk(expand(P), z...)

        function $lk_layout(LAY::ApplyLayout{typeof(*)}, V::AbstractQuasiVecOrMat, y...)
            a = arguments(LAY, V)
            *($lk(a[1], y...), tail(a)...)
        end

        $lk_layout(::ExpansionLayout, A, dims...) = $lk_layout(ApplyLayout{typeof(*)}(), A, dims...)
        $lk_layout(::SubBasisLayout, A, dims...) = $lk(parent(A), dims...)[:, parentindices(A)[2]]
    end
end


end # module SingularIntegrals
