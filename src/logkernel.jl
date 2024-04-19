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
function logkernel_layout(::Union{MappedBasisLayouts, MappedOPLayouts}, wT)
    V = eltype(wT)
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

function logkernel_demap(wT, z)
    P = demap(wT)
    kr = basismap(wT)
    z̃ = inbounds_getindex(kr, z)
    c = inv(kr.A)
    LP = logkernel(P, z̃)
    Σ = sum(P; dims=1)
    transpose(c*transpose(LP) + c*log(c)*vec(Σ))
end


logkernel_layout(::Union{MappedBasisLayouts, MappedOPLayouts}, wT, z::Number) = logkernel_demap(wT, z)
logkernel_layout(::WeightedOPLayout{MappedOPLayout}, wT, z::Real) = logkernel_demap(wT, z)





####
# LogKernelPoint
####

function complexlogkernel(wP::Weighted{<:Any,<:ChebyshevU}, z::Number)
    T = promote_type(eltype(wP), typeof(z))
    ξ = inv(z + sqrtx2(z))
    r0 = convert(T,π)*(ξ^2/4 - (log.(abs.(ξ)) + log(2*one(T)))/2)
    r1 = convert(T,π)*(ξ^3/3 - ξ)/2
    r2 = convert(T,π)*(ξ^4/4 - ξ^2/2)/2
    # We have for n ≠ 0 L_n(z) = stieltjes(U_n, z)
    # We have diff( /2
    # where U_n(z) = ∫_1^x sqrt(1-x^2) U_n(x) dx = -(1-x^2)^(3/2)C_{n-1}(x) * 2/(n * (n + 2))
    # where C_n(x) = C_n^{(2)}(x)
    # We have the 3-term recurrence
    # C_{n+1}(x) == 2(n + 2) / (n + 1) * x C_n(x) - (n + 3) / (n + 1) C_{n-1}(x)
    # Thus
    # U_{n+1}(x) == -(1-x^2)^(3/2) C_n(x) * 2/((n+1) * (n + 3))
    # == -(1-x^2)^(3/2) * 2/((n+1) * (n + 3)) ( 2(n + 1) / n  * x C_{n-1}(x) - (n + 2) / n C_{n-2}(x))
    # == -(1-x^2)^(3/2) *  (4 / (n*(n+3))  * x C_{n-1}(x) - 2 (n + 2) /(n*(n+1)*(n+3)) C_{n-2}(x))
    # == -(1-x^2)^(3/2) *  (2 (n-1)*(n+1) / (n*(n+3))  * x  2/((n-1)*(n+1)) C_{n-1}(x) -  (n + 2)*(n-1) /(n*(n+3)) * 2/((n-1)*(n+1)) C_{n-2}(x))
    # == (2 (n+2)/(n+3)  * x  U_n(x) -  (n + 2)*(n-1) /(n*(n+3)) * U_{n-1}(x)

    n = zero(real(T)):∞
    A = (2*(n .+ 2)) ./ (n .+ 3)
    B = Zeros{real(T)}(∞)
    C =  (n .+ 2) .* (n .- 1) ./ (n .* (n .+ 3))
    transpose(RecurrenceArray(z, (A, B, C), [r0,r1,r2]))
end

function complexlogkernel(P::Weighted{<:Any,<:Legendre}, z::Number)
    T = promote_type(eltype(P), typeof(z))
    r0 = (1 + z)log(1 + z) - (z-1)log(z-1) - 2one(T)
    r1 = (z+1)*r0/2 + 1 - (z+1)log(z+1)
    r2 = z*r1 + 2*one(T)/3
    # We have for n ≠ 0 L_n(z) = stieltjes(U_n, z)
    # where U_n(z) = ∫_1^x P_n(x) dx = C_{n+1}^{(-1/2)}(x)
    # Since these are equivalent to weihted OPs (1-x^2)C_{n-1}^(3/2)(x)
    # we know they satisfy the same recurrence coefficients.
    # Thus the following could also be written:
    # A,B,C = recurrencecoefficients(Ultraspherical(-1/2))
    # A[2:end],B[2:end],C[2:end]

    A,B,C = ((one(real(T)):2:∞)./(2:∞), Zeros{real(T)}(∞), (-one(real(T)):∞)./(2:∞))
    transpose(RecurrenceArray(z, (A, B, C), [r0,r1,r2]))
end

function complexlogkernel(P::Weighted{<:Any,<:Legendre}, zs::AbstractVector)
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

complexlogkernel(P::Legendre, z...) = complexlogkernel(Weighted(P), z...)


logkernel_layout(::AbstractBasisLayout, P, z...) = real.(complexlogkernel(P, z...))

function logkernel_layout(::WeightedOPLayout, P, x::Real)
    L = transpose(complexlogkernel(P, complex(x)))
    transpose(RecurrenceArray(x, (L.A, L.B, L.C), real.(L.data)))
end

logkernel(P::Legendre, x...) = logkernel(Weighted(P), x...)

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


