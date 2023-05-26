const LogKernelPoint{T<:Real,C,W<:Number,V,D} = BroadcastQuasiMatrix{T,typeof(log),Tuple{BroadcastQuasiMatrix{T,typeof(abs),Tuple{ConvKernel{C,W,V,D}}}}}
const LogKernel{T,D1,D2} = BroadcastQuasiMatrix{T,typeof(log),Tuple{BroadcastQuasiMatrix{T,typeof(abs),Tuple{ConvKernel{T,Inclusion{T,D1},T,D2}}}}}


###
# LogKernel
###

@simplify function *(L::LogKernel{<:Any,<:ChebyshevInterval,<:ChebyshevInterval}, wT::Weighted{<:Any,<:ChebyshevT})
    T = promote_type(eltype(L), eltype(wT))
    ChebyshevT{T}() * Diagonal(Vcat(-convert(T,π)*log(2*one(T)),-convert(T,π)./(1:∞)))
end

@simplify function *(H::LogKernel, wT::Weighted{<:Any,<:SubQuasiArray{<:Any,2,<:ChebyshevT,<:Tuple{AbstractAffineQuasiVector,Slice}}})
    V = promote_type(eltype(H), eltype(wT))
    kr,jr = parentindices(wT.P)
    @assert axes(H,1) == axes(H,2) == axes(wT,1)
    T = parent(wT.P)
    x = axes(T,1)
    W = Weighted(T)
    A = kr.A
    T[kr,:] * Diagonal(Vcat(-convert(V,π)*(log(2*one(V))+log(abs(A)))/A,-convert(V,π)./(A * (1:∞))))
end



####
# LogKernelPoint
####

@simplify function *(L::LogKernelPoint, wP::Weighted{<:Any,<:ChebyshevU})
    T = promote_type(eltype(L), eltype(wP))
    z, xc = parent(L).args[1].args[1].args
    if z in axes(wP,1)
        Tn = Vcat(convert(T,π)*log(2*one(T)), convert(T,π)*ChebyshevT{T}()[z,2:end]./oneto(∞))
        return transpose((Tn[3:end]-Tn[1:end])/2)
    else
        # for U_k where k>=1
        ξ = inv(z + sqrtx2(z))
        ζ = (convert(T,π)*ξ.^oneto(∞))./oneto(∞)
        ζ = (ζ[3:end]- ζ[1:end])/2

        # for U_0
        ζ = Vcat(convert(T,π)*(ξ^2/4 - (log.(abs.(ξ)) + log(2*one(T)))/2), ζ)
        return transpose(ζ)
    end

end

@simplify function *(L::LogKernelPoint, P::Legendre)
    T = promote_type(eltype(L), eltype(P))
    z, xc = parent(L).args[1].args[1].args
    @assert z > 1
    r0 = (1 + z)log(1 + z) - (z-1)log(z-1) - 2one(T)
    r1 = (z+1)*r0/2 + 1 - (z+1)log(z+1)
    r2 = z*r1 + 2*one(T)/3
    transpose(RecurrenceArray(z, ((one(T):2:∞)./(2:∞), Zeros{T}(∞), (-one(T):∞)./(2:∞)), [r0,r1,r2]))
end


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


