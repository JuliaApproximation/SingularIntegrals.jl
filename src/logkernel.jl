const ComplexLogKernelPoint{T,C,W<:Number,V,D} = BroadcastQuasiMatrix{T,typeof(log),Tuple{ConvKernel{C,W,V,D}}}
const LogKernelPoint{T<:Real,C,W<:Number,V,D} = BroadcastQuasiMatrix{T,typeof(log),Tuple{BroadcastQuasiMatrix{T,typeof(abs),Tuple{ConvKernel{C,W,V,D}}}}}
const LogKernel{T,D1,D2} = BroadcastQuasiMatrix{T,typeof(log),Tuple{BroadcastQuasiMatrix{T,typeof(abs),Tuple{ConvKernel{T,Inclusion{T,D1},T,D2}}}}}


@simplify function *(L::LogKernel, P::AbstractQuasiVecOrMat)
    T = promote_type(eltype(L), eltype(P))
    logkernel(convert(AbstractQuasiArray{T}, P))
end

@simplify function *(L::LogKernelPoint, P::AbstractQuasiVecOrMat)
    T = promote_type(eltype(L), eltype(P))
    z, xc = L.args[1].args[1].args
    logkernel(convert(AbstractQuasiArray{T}, P), z)
end

@simplify function *(L::ComplexLogKernelPoint, P::AbstractQuasiVecOrMat)
    z, xc = L.args[1].args
    T = promote_type(eltype(L), eltype(P))
    complexlogkernel(convert(AbstractQuasiArray{T}, P), z)
end

###
# LogKernel
###

"""
    logkernel(P)

applies the log kernel log(abs(x-t)) to the columns of a quasi matrix, i.e., `(log.(abs.(x - x')) * P)/π`
"""
logkernel(P, z...) = logkernel_layout(MemoryLayout(P), P, z...)



"""
    complexlogkernel(P)

applies the log kernel log(x-t) to the columns of a quasi matrix, i.e., `(log.(x - x') * P)`
"""
complexlogkernel(P, z...) = complexlogkernel_layout(MemoryLayout(P), P, z...)

logkernel(wT::Weighted{T,<:ChebyshevT}) where T = ChebyshevT{T}() * Diagonal(Vcat(-convert(T,π)*log(2*one(T)),-convert(T,π)./(1:∞)))
function logkernel_layout(::Union{MappedBasisLayouts, MappedOPLayouts}, wT::AbstractQuasiMatrix{V}) where V
    kr = basismap(wT)
    @assert kr isa AbstractAffineQuasiVector
    W = demap(wT)
    A = kr.A
    # L = logkernel(W)
    # Σ = sum(W; dims=1)
    # basis(L)[kr,:] * (coefficients(L)/A - OneElement(∞) * Σ*log(abs(A))/A)
    @assert W isa Weighted{<:Any,<:ChebyshevT}
    unweighted(wT) * Diagonal(Vcat(-convert(V,π)*(log(2*one(V))+log(abs(A)))/A,-convert(V,π) ./ (A * (1:∞))))
end


function logkernel_layout(::Union{MappedBasisLayouts, MappedOPLayouts}, wT::AbstractQuasiMatrix{V}, z::Number) where V
    P = demap(wT)
    kr = basismap(wT)
    z̃ = inbounds_getindex(kr, z)
    c = inv(kr.A)
    LP = logkernel(P, z̃)
    Σ = sum(P; dims=1)
    transpose(c*transpose(LP) + c*log(c)*vec(Σ))
end






####
# LogKernelPoint
####


function complexlogkernel(wP::Weighted{T,<:ChebyshevU}, z::Number) where T
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

function complexlogkernel(P::Legendre, z::Number)
    T = promote_type(eltype(P), typeof(z))
    r0 = (1 + z)log(1 + z) - (z-1)log(z-1) - 2one(T)
    r1 = (z+1)*r0/2 + 1 - (z+1)log(z+1)
    r2 = z*r1 + 2*one(T)/3
    transpose(RecurrenceArray(z, ((one(real(T)):2:∞)./(2:∞), Zeros{real(T)}(∞), (-one(real(T)):∞)./(2:∞)), [r0,r1,r2]))
end

function complexlogkernel(P::Legendre, zs::AbstractVector)
    T = promote_type(eltype(P), eltype(zs))
    m = length(zs)
    data = Matrix{T}(undef, 3, m)
    for j = 1:m
        z = zs[j]
        r0 = (1 + z)log(1 + z) - (z-1)log(z-1) - 2one(T)
        r1 = (z+1)*r0/2 + 1 - (z+1)log(z+1)
        r2 = z*r1 + 2*one(T)/3
        data[1,j] = r0; data[2,j] = r1; data[3,j] = r2;
    end

    transpose(RecurrenceArray(zs, ((one(real(T)):2:∞)./(2:∞), Zeros{real(T)}(∞), (-one(real(T)):∞)./(2:∞)), data))
end


logkernel(P::Legendre, z) = real.(complexlogkernel(P, complex(z)))

function logkernel(P::Legendre, x::Real)
    T = promote_type(eltype(P), typeof(x))
    z = complex(x)
    r0 = (1 + z)log(1 + z) - (z-1)log(z-1) - 2one(T)
    r1 = (z+1)*r0/2 + 1 - (z+1)log(z+1)
    r2 = z*r1 + 2*one(T)/3
    transpose(RecurrenceArray(x, ((one(real(T)):2:∞)./(2:∞), Zeros{real(T)}(∞), (-one(real(T)):∞)./(2:∞)), [real(r0),real(r1),real(r2)]))
end

function logkernel(P::Legendre, x::AbstractVector{<:Real})
    T = promote_type(eltype(P), eltype(x))
    m = length(x)
    data = Matrix{T}(undef, 3, m)
    for j = 1:m
        z = complex(x[j])
        r0 = (1 + z)log(1 + z) - (z-1)log(z-1) - 2one(T)
        r1 = (z+1)*r0/2 + 1 - (z+1)log(z+1)
        r2 = z*r1 + 2*one(T)/3
        data[1,j] = real(r0); data[2,j] = real(r1); data[3,j] = real(r2);
    end

    transpose(RecurrenceArray(x, ((one(real(T)):2:∞)./(2:∞), Zeros{real(T)}(∞), (-one(real(T)):∞)./(2:∞)), data))
end


###
# Maps
###



function logkernel(S::PiecewiseInterlace, z::Number)
    @assert length(S.args) == 2
    a,b = S.args
    xa,xb = axes(a,1),axes(b,1)
    Sa = logkernel(a, z)
    Sb = logkernel(b, z)
    transpose(BlockBroadcastArray(vcat, unitblocks(transpose(Sa)), unitblocks(transpose(Sb))))
end


