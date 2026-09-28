module SingularIntegrals
using ClassicalOrthogonalPolynomials, ContinuumArrays, QuasiArrays, LazyArrays, LazyBandedMatrices, FillArrays, BandedMatrices, LinearAlgebra, SpecialFunctions, HypergeometricFunctions, InfiniteArrays
using ContinuumArrays: @simplify, Weight, AbstractAffineQuasiVector, inbounds_getindex, broadcastbasis, MappedBasisLayouts, MemoryLayout, MappedWeightLayout, AbstractWeightLayout, ExpansionLayout, demap, basismap, AbstractBasisLayout, SubBasisLayout, PiecewiseBasis
using QuasiArrays: AbstractQuasiMatrix, BroadcastQuasiMatrix, LazyQuasiArrayStyle, AbstractQuasiVecOrMat
import ClassicalOrthogonalPolynomials: AbstractJacobiWeight, AbstractJacobi, WeightedBasis, jacobimatrix, orthogonalityweight, recurrencecoefficients, _p0, chop, initiateforwardrecurrence, MappedOPLayouts, unweighted, WeightedOPLayout, MappedOPLayout, SetindexInterlace, interlace_setindex
using LazyBandedMatrices: Tridiagonal, SymTridiagonal, subdiagonaldata, supdiagonaldata, diagonaldata, ApplyLayout
import LazyArrays: AbstractCachedMatrix, AbstractCachedArray, paddeddata, arguments, resizedata!, cache_filldata!, zero!, cacheddata, LazyArrayStyle
import Base: *, +, -, /, \, Slice, axes, getindex, sum, ==, oneto, size, broadcasted, copy, tail, view
import LinearAlgebra: dot
using BandedMatrices: _BandedMatrix
using RecurrenceRelationshipArrays
using RecurrenceRelationshipArrays: Clenshaw

export associated, stieltjes, cauchy, hilbert, logkernel, powerkernel, complexlogkernel


include("stieltjes.jl")
include("logkernel.jl")
include("power.jl")


### generic fallback
for Op in (:PVStieltjes, :Stieltjes, :StieltjesPoint, :LogKernelPoint, :PowerKernelPoint, :LogKernel)
    @eval begin
        @simplify function *(H::$Op, wP::WeightedBasis{<:Any,<:Weight,<:Any})
            w,P = wP.args
            Q = OrthogonalPolynomial(w)
            (H * Weighted(Q)) * (Q \ P)
        end
        @simplify *(H::$Op, wP::Weighted{<:Any,<:SubQuasiArray{<:Any,2}}) = H * view(Weighted(parent(wP.P)), parentindices(wP.P)...)
    end
end


for lk in (:complexlogkernel, :stieltjes)
    lk_layout = Symbol(lk, :_layout)
    @eval $lk_layout(::AbstractBasisLayout, P, z...) = error("not implemented")
end

# general routines
for lk in (:hilbert, :logkernel, :complexlogkernel, :stieltjes)
    lk_layout = Symbol(lk, :_layout)
    @eval begin
        $lk_layout(::AbstractWeightLayout, w, zs::AbstractVector) = [$lk(w, z) for z in zs]
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

        # only is needed for array-valued bases which return a 1×∞ matrix as transpose is recursive
        function $lk_layout(LAY::ApplyLayout{typeof(*)}, V::AbstractQuasiVector, z::Number)
            a = arguments(LAY, V)
            only(*($lk(a[1], z), tail(a)...))
        end

        $lk_layout(::ExpansionLayout, A, dims...) = $lk_layout(ApplyLayout{typeof(*)}(), A, dims...)
        $lk_layout(::SubBasisLayout, A, dims...) = $lk(parent(A), dims...)[:, parentindices(A)[2]]
    end
end

# array-valued bases
for lk in (:hilbert, :logkernel, :complexlogkernel, :stieltjes)
    @eval function $lk(S::SetindexInterlace, z::Number)
        Ls = map(a -> $lk(a, z), S.args)
        z̃ = S.z .+ zero(mapreduce(eltype, promote_type, Ls)) # promote, e.g., to complex
        # use hcat as transpose is recursive
        BlockBroadcastArray{typeof(z̃)}(hcat, map((i,L) -> unitblocks(interlace_setindex.(Ref(z̃), L, i)), Base.OneTo(length(Ls)), Ls)...)
    end
end


end # module SingularIntegrals
