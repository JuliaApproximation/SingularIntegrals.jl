
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


