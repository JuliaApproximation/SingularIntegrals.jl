const ComplexLogKernelPoint{T,C,W<:Number,V,D} = BroadcastQuasiMatrix{T,typeof(log),Tuple{ConvKernel{C,W,V,D}}}
const LogKernelPoint{T<:Real,C,W<:Number,V,D} = BroadcastQuasiMatrix{T,typeof(log),Tuple{BroadcastQuasiMatrix{T,typeof(abs),Tuple{ConvKernel{C,W,V,D}}}}}
const LogKernel{T,D1,D2} = BroadcastQuasiMatrix{T,typeof(log),Tuple{BroadcastQuasiMatrix{T,typeof(abs),Tuple{ConvKernel{T,Inclusion{T,D1},T,D2}}}}}


###
# LogKernel
###

@simplify *(L::LogKernel, P::AbstractQuasiMatrix) = logkernel(P)*π

"""
    logkernel(P)

applies the log kernel log(x-t)/π to the columns of a quasi matrix, i.e., `(log.(x - x') * P)/π`
"""
logkernel(P, z...) = logkernel_layout(MemoryLayout(P), P, z...)
logkernel_layout(lay, P, z...) = error("not implemented")

logkernel(wT::Weighted{T,<:ChebyshevT}) where T = ChebyshevT{T}() * Diagonal(Vcat(-log(2*one(T)),inv.(-(1:∞))))
function logkernel_layout(::MappedBasisLayout, wT::SubQuasiArray{V,2}) where V
    kr,jr = parentindices(wT.P)
    @assert kr isa AbstractAffineQuasiVector
    T = parent(wT.P)
    x = axes(T,1)
    W = Weighted(T)
    A = kr.A
    T[kr,:] * Diagonal(Vcat(-(log(2*one(V))+log(abs(A)))/A,-inv.(A * (1:∞))))
end



####
# LogKernelPoint
####

@simplify function *(L::LogKernelPoint, P::AbstractQuasiMatrix)
    z, xc = parent(L).args[1].args[1].args
    logkernel(P, z)*π
end

@simplify function *(L::ComplexLogKernelPoint, P::AbstractQuasiMatrix)
    z, xc = parent(L).args[1].args[1].args
    complexlogkernel(P, z)*π
end

function logkernel(wP::Weighted{T,<:ChebyshevU}, z) where T
    if z in axes(wP,1)
        Tn = Vcat(log(2*one(T)), ChebyshevT{T}()[z,2:end]./oneto(∞))
        return transpose((Tn[3:end]-Tn[1:end])/2)
    else
        # for U_k where k>=1
        ξ = inv(z + sqrtx2(z))
        ζ = ξ.^oneto(∞) ./ oneto(∞)
        ζ = (ζ[3:end]- ζ[1:end])/2

        # for U_0
        ζ = Vcat(ξ^2/4 - (log.(abs.(ξ)) + log(2*one(T)))/2, ζ)
        return transpose(ζ)
    end

end

@simplify function complexlogkernel(P::Legendre{T}, z) where T
    r0 = (1 + z)log(1 + z) - (z-1)log(z-1) - 2one(T)
    r1 = (z+1)*r0/2 + 1 - (z+1)log(z+1)
    r2 = z*r1 + 2*one(T)/3
    transpose(RecurrenceArray(z, ((one(real(T)):2:∞)./(2:∞), Zeros{real(T)}(∞), (-one(real(T)):∞)./(2:∞)), [r0,r1,r2]))
end

@simplify function *(L::LogKernelPoint, P::Legendre)
    T = promote_type(eltype(L), eltype(P))
    z, x = parent(L).args[1].args[1].args
    real.(log.(complex(z) .- x) * P)
end


###
# Maps
###


@simplify function *(L::LogKernelPoint, wT::SubQuasiArray{<:Any,2,<:Any,<:Tuple{<:AbstractAffineQuasiVector,<:Any}})
    P = parent(wT)
    z, xc = parent(L).args[1].args[1].args
    kr, jr = parentindices(wT)
    z̃ = inbounds_getindex(kr, z)
    x̃ = axes(P,1)
    c = inv(kr.A)
    LP = log.(abs.(z̃ .- x̃')) * P
    Σ = sum(P; dims=1)
    transpose((c*transpose(LP) + c*log(c)*vec(Σ))[jr])
end

@simplify function *(L::LogKernel, wT::SubQuasiArray{<:Any,2,<:Any,<:Tuple{<:AbstractAffineQuasiVector,<:Slice}})
    V = promote_type(eltype(L), eltype(wT))
    wP = parent(wT)
    kr, jr = parentindices(wT)
    x = axes(wP,1)
    T = wP.P
    @assert T isa ChebyshevT
    D = T \ (log.(abs.(x .- x')) * wP)
    c = inv(2*kr.A)
    T[kr,:] * Diagonal(Vcat(2*convert(V,π)*c*log(c), 2c*D.diag.args[2]))
end

@simplify function *(L::LogKernelPoint, S::PiecewiseInterlace)
    z, xc = parent(L).args[1].args[1].args
    axes(L,2) == axes(S,1) || throw(DimensionMismatch())
    @assert length(S.args) == 2
    a,b = S.args
    xa,xb = axes(a,1),axes(b,1)
    Sa = log.(abs.(z .- xa')) * a
    Sb = log.(abs.(z .- xb')) * b
    transpose(BlockBroadcastArray(vcat, unitblocks(transpose(Sa)), unitblocks(transpose(Sb))))
end


