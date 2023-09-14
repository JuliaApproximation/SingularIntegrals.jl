####
# Associated
####


"""
    AssociatedWeighted(P)

We normalise so that `orthogonalityweight(::Associated)` is a probability measure.
"""
struct AssociatedWeight{T,OPs<:AbstractQuasiMatrix{T}} <: Weight{T}
    P::OPs
end
axes(w::AssociatedWeight) = (axes(w.P,1),)

sum(::AssociatedWeight{T}) where T = one(T)

"""
    Associated(P)

constructs the associated orthogonal polynomials for P, which have the Jacobi matrix

    jacobimatrix(P)[2:end,2:end]

and constant first term. Or alternatively

    w = orthogonalityweight(P)
    A = recurrencecoefficients(P)[1]
    Associated(P) == (w/(sum(w)*A[1]))'*((P[:,2:end]' - P[:,2:end]) ./ (x' - x))

where `x = axes(P,1)`.
"""

struct Associated{T, OPs<:AbstractQuasiMatrix{T}} <: OrthogonalPolynomial{T}
    P::OPs
end

associated(P) = Associated(P)

axes(Q::Associated) = axes(Q.P)
==(A::Associated, B::Associated) = A.P == B.P

orthogonalityweight(Q::Associated) = AssociatedWeight(Q.P)

function associated_jacobimatrix(X::Tridiagonal)
    c,a,b = subdiagonaldata(X),diagonaldata(X),supdiagonaldata(X)
    Tridiagonal(c[2:end], a[2:end], b[2:end])
end

function associated_jacobimatrix(X::SymTridiagonal)
    a,b = diagonaldata(X),supdiagonaldata(X)
    SymTridiagonal(a[2:end], b[2:end])
end
jacobimatrix(a::Associated) = associated_jacobimatrix(jacobimatrix(a.P))

associated(::ChebyshevT{T}) where T = ChebyshevU{T}()
associated(::ChebyshevU{T}) where T = ChebyshevU{T}()


const ConvKernel{T,D1,V,D2} = BroadcastQuasiMatrix{T,typeof(-),Tuple{D1,QuasiAdjoint{V,Inclusion{V,D2}}}}
const StieltjesPoint{T,W<:Number,V,D} = BroadcastQuasiMatrix{T,typeof(inv),Tuple{ConvKernel{T,W,V,D}}}
const StieltjesPoints{T,W<:AbstractVector{<:Number},V,D} = BroadcastQuasiMatrix{T,typeof(inv),Tuple{ConvKernel{T,W,V,D}}}
const Stieltjes{T,D1,D2} = BroadcastQuasiMatrix{T,typeof(inv),Tuple{ConvKernel{T,Inclusion{T,D1},T,D2}}}



@simplify function *(H::Stieltjes, w::AbstractQuasiVecOrMat)
    T = promote_type(eltype(H), eltype(w))
    stieltjes(convert(AbstractQuasiArray{T}, w), axes(H,1))
end

@simplify function *(H::StieltjesPoint, w::AbstractQuasiMatrix)
    T = promote_type(eltype(H), eltype(w))
    z = H.args[1].args[1]
    convert(AbstractArray{T}, stieltjes(w, z))
end

@simplify function *(H::StieltjesPoint, w::AbstractQuasiVector)
    T = promote_type(eltype(H), eltype(w))
    z = H.args[1].args[1]
    convert(T, stieltjes(w, z))
end

"""
    stieltjes(P, y)

computes inv.(y - x') * P understood in a principle value sense.
"""
stieltjes(P, y...) = stieltjes_layout(MemoryLayout(P), P, y...)

"""
    stieltjes(P)

computes inv.(x - x') * P understood in a principle value sense.
"""
stieltjes(w::ChebyshevTWeight{T}) where T = zeros(T, axes(w,1))
stieltjes(w::ChebyshevUWeight{T}) where T = convert(T,π) * axes(w,1)
function stieltjes(w::LegendreWeight{T}) where T
    x = axes(w,1)
    log.(x .+ one(T)) .- log.(one(T) .- x)
end


stieltjes(wT::Weighted{T,<:ChebyshevT}) where T = ChebyshevU{T}() * _BandedMatrix(Fill(-convert(T,π),1,∞), ℵ₀, -1, 1)
stieltjes(wU::Weighted{T,<:ChebyshevU}) where T = ChebyshevT{T}() * _BandedMatrix(Fill(convert(T,π),1,∞), ℵ₀, 1, -1)



function stieltjes(wP::Weighted{<:Any,<:OrthogonalPolynomial})
    P = wP.P
    w = orthogonalityweight(P)
    A = recurrencecoefficients(P)[1]
    Q = associated(P)
    (-A[1]*sum(w))*[zero(axes(P,1)) Q] + stieltjes(w) .* P
end

stieltjes(P::Legendre) = stieltjes(Weighted(P))


##
# OffStieltjes
##

function stieltjes(W::Weighted{<:Any,<:ChebyshevU}, x::Inclusion)
    x == axes(W,1) && return stieltjes(W)
    tol = eps()
    T̃ = chebyshevt(x)
    ψ_1 = T̃ \ inv.(x .+ sqrtx2.(x)) # same ψ_1 = x .- sqrt(x^2 - 1) but with relative accuracy as x -> ∞
    M = Clenshaw(T̃ * ψ_1, T̃)
    data = zeros(eltype(ψ_1), ∞, ∞)
    # Operator has columns π * ψ_1^k
    copyto!(view(data,:,1), convert(eltype(data),π)*ψ_1)
    for j = 2:∞
        mul!(view(data,:,j),M,view(data,:,j-1))
        norm(view(data,:,j)) ≤ tol && break
    end
    # we wrap in a Padded to avoid increasing cache size
    T̃ * PaddedArray(chop(paddeddata(data), tol), size(data)...)
end





####
# StieltjesPoint
####

stieltjesmoment_jacobi_normalization(n::Int,α::Real,β::Real) = 2^(α+β)*gamma(n+α+1)*gamma(n+β+1)/gamma(2n+α+β+2)

function stieltjes(w::AbstractJacobiWeight, z::Number)
    α,β = real(w.a),real(w.b)
    (x = 2/(1-z);stieltjesmoment_jacobi_normalization(0,α,β)*HypergeometricFunctions.mxa_₂F₁(1,α+1,α+β+2,x))
end

function stieltjes(w::ChebyshevTWeight{T}, z::Number) where T
    α,β = w.a,w.b
    z in axes(w,1) && return zero(T)
    convert(T, π)/sqrtx2(z)
end

function stieltjes(w::ChebyshevUWeight{T}, z::Number) where T
    α,β = w.a,w.b
    z in axes(w,1) && return π*z
    convert(T, π)/(z + sqrtx2(z))
end

@simplify function *(S::StieltjesPoints, w::Weight)
    zs = S.args[1].args[1] # vector of points to eval at
    stieltjes(w, zs)
end

function stieltjes(wP::Weighted, z::Number)
    P = wP.P
    w = orthogonalityweight(P)
    A,B,C = recurrencecoefficients(P)
    r1 = stieltjes(w, z)*_p0(P) # stieltjes of the weight
    # (a[1]-z)*r[1] + b[1]r[2] == -sum(w)*_p0(P)
    # (a[1]/b[1]-z/b[1])*r[1] + r[2] == -sum(w)*_p0(P)/b[1]
    # (A[1]z + B[1])*r[1] - r[2] == A[1]sum(w)*_p0(P)
    # (A[1]z + B[1])*r[1]-A[1]sum(w)*_p0(P) ==  r[2] 
    r2 = (A[1]z + B[1])*r1-A[1]sum(w)*_p0(P)
    transpose(RecurrenceArray(z, (A,B,C), [r1,r2]))
end

function stieltjes(wP::Weighted, z::AbstractVector)
    T = promote_type(eltype(z), eltype(wP))
    P = wP.P
    A,B,C = recurrencecoefficients(P)
    w = orthogonalityweight(P)
    data = Matrix{T}(undef, 2, length(z))
    data[1,:] .= stieltjes(w, z) .* _p0(P)
    data[2,:] .= (A[1] .* z .+ B[1]) .* data[1,:] .- (A[1]sum(w)*_p0(P))
    transpose(RecurrenceArray(z, (A,B,C), data))
end

sqrtx2(z::Number) = sqrt(z-1)*sqrt(z+1)
sqrtx2(x::Real) = sign(x)*sqrt(x^2-1)


stieltjes(P::Legendre, z...) = stieltjes(Weighted(P), z...)

@simplify function *(S::StieltjesPoints, wP::Weighted)
    z = S.args[1].args[1] # vector of points to eval at
    stieltjes(wP, z)
end

@simplify function *(S::StieltjesPoints, P::Legendre)
    S * Weighted(P)
end



##
# mapped
###

function stieltjes_layout(::MappedWeightLayout, w::SubQuasiArray{<:Any,1})
    m = parentindices(w)[1]
    # TODO: mapping other geometries
    P = parent(w)
    stieltjes(P)[m]
end

function stieltjes_layout(::MappedWeightLayout, w::AbstractQuasiVector, z::Number)
    m = basismap(w)
    # TODO: mapping other geometries
    P = demap(w)
    stieltjes(P, inbounds_getindex(m, z))
end

function stieltjes_layout(::Union{MappedBasisLayouts, MappedOPLayouts}, wP::AbstractQuasiMatrix, x::Inclusion)
    kr = basismap(wP)
    W = demap(wP)
    t̃ = axes(W,1)
    t = axes(wP,1)

    x == t && return stieltjes(W)[kr,:]

    M = affine(t,t̃)
    @assert x isa Inclusion
    a,b = first(x),last(x)
    x̃ = Inclusion((M.A * a .+ M.b)..(M.A * b .+ M.b)) # map interval to new interval
    Q̃,M = arguments(*, stieltjes(W, x̃))
    parent(Q̃)[affine(x,axes(parent(Q̃),1)),:] * M
end

function stieltjes_layout(::Union{MappedBasisLayouts, MappedOPLayouts}, wT::AbstractQuasiMatrix, z::Number)
    P = demap(wT)
    z̃ = inbounds_getindex(basismap(wT), z)
    stieltjes(P, z̃)
end

###
# Interlace
###


function stieltjes(W::PiecewiseInterlace)
    Hs = broadcast(function(a,b)
                x,t = axes(a,1),axes(b,1)
                H = stieltjes(b, x)
                H
            end, [W.args...], permutedims([W.args...]))
    N = length(W.args)
    Ts = [broadcastbasis(+, broadcast(H -> H.args[1], Hs[k,:])...) for k=1:N]
    Ms = broadcast((T,H) -> unitblocks(T\H), Ts, Hs)
    PiecewiseInterlace(Ts...) * BlockBroadcastArray{eltype(W)}(hvcat, N, permutedims(Ms)...)
end


function stieltjes(S::PiecewiseInterlace, z::Number)
    @assert length(S.args) == 2
    a,b = S.args
    Sa = stieltjes(a, z)
    Sb = stieltjes(b, z)
    transpose(BlockBroadcastArray(vcat, unitblocks(transpose(Sa)), unitblocks(transpose(Sb))))
end