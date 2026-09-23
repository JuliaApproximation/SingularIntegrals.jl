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
const PVStieltjes{T,D1} = BroadcastQuasiMatrix{T,typeof(pinv),Tuple{ConvKernel{T,Inclusion{T,D1},T,D1}}}
const PVStieltjesPoint{T,W<:Number,V,D} = BroadcastQuasiMatrix{T,typeof(pinv),Tuple{ConvKernel{T,W,V,D}}}
const PVStieltjesPoints{T,W<:AbstractVector{<:Number},V,D} = BroadcastQuasiMatrix{T,typeof(pinv),Tuple{ConvKernel{T,W,V,D}}}




@simplify function *(H::PVStieltjes, w::AbstractQuasiVecOrMat)
    T = promote_type(eltype(H), eltype(w))
    convert(T,π)*hilbert(convert(AbstractQuasiArray{T}, w))
end

@simplify function *(H::Stieltjes, w::AbstractQuasiMatrix)
    T = promote_type(eltype(H), eltype(w))
    z = H.args[1].args[1]
    convert(AbstractQuasiArray{T}, stieltjes(w, z))
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

@simplify function *(H::PVStieltjesPoint, w::AbstractQuasiMatrix)
    T = promote_type(eltype(H), eltype(w))
    z = H.args[1].args[1]
    convert(AbstractArray{T}, hilbert(w, z)*π)
end

@simplify function *(H::PVStieltjesPoint, w::AbstractQuasiVector)
    T = promote_type(eltype(H), eltype(w))
    z = H.args[1].args[1]
    convert(T, hilbert(w, z)*π)
end


"""
    stieltjes(P, z)

computes inv.(z - t') * P where t = axes(P,1).
"""
stieltjes(P, y...) = stieltjes_layout(MemoryLayout(P), P, y...)

"""
    cauchy(P, z)

computes inv.(t'-z) * P/(2π*im) where t = axes(P,1).
"""
cauchy(f, z...) = stieltjes(f, z...)/(-2convert(eltype(f), π)*im)

"""
    hilbert(P, x)

computes inv.(x - t') * P/π in a principle value sense where t = axes(P,1) and x in t.
"""
hilbert(P, x...) = hilbert_layout(MemoryLayout(P), P, x...)

"""
    hilbert(P)

computes (inv.(x - x') * P)/π understood in a principle value sense.
"""
hilbert(w::ChebyshevTWeight{T}) where T = zeros(T, axes(w,1))
hilbert(w::ChebyshevUWeight{T}) where T = axes(w,1)
hilbert(w::Weight) = hilbert.(Ref(w), axes(w,1)) # use pointwise


hilbert(wT::Weighted{T,<:ChebyshevT}) where T = ChebyshevU{T}() * _BandedMatrix(-Ones{T}(1,∞), ℵ₀, -1, 1)
hilbert(wU::Weighted{T,<:ChebyshevU}) where T = ChebyshevT{T}() * _BandedMatrix(Ones{T}(1,∞), ℵ₀, 1, -1)



function hilbert(wP::Weighted{<:Any,<:OrthogonalPolynomial})
    P = wP.P
    w = orthogonalityweight(P)
    A = recurrencecoefficients(P)[1]
    Q = associated(P)
    (-A[1]*sum(w)/π)*[zero(axes(P,1)) Q] + hilbert(w) .* P
end


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
    x = 2/(1-z)
    stieltjesmoment_jacobi_normalization(0,α,β)*HypergeometricFunctions.mxa_₂F₁(1,α+1,α+β+2,x)
end

stieltjes(w::ChebyshevTWeight{T}, z::Number) where T = convert(T, π)/sqrtx2(z)

function hilbert(w::ChebyshevTWeight{T}, x::Number) where T
    x in axes(w,1) || throw(DomainError(x))
    return zero(T)
end

function hilbert(w::ChebyshevUWeight{T}, x::Number) where T
    x in axes(w,1) || throw(DomainError(x))
    return convert(T, x)
end

function hilbert(w::LegendreWeight{T}, x::Number) where T
    x in axes(w,1) || throw(DomainError(x))
    (log(x + one(T)) - log(one(T) - x))/π
end


stieltjes(w::ChebyshevUWeight{T}, z::Number) where T = convert(T, π)/(z + sqrtx2(z))

@simplify function *(S::StieltjesPoints, w::Weight)
    zs = S.args[1].args[1] # vector of points to eval at
    stieltjes(w, zs)
end

_stielsum(::typeof(stieltjes), f) = sum(f)
_stielsum(::typeof(hilbert), f) = sum(f)/π

for stiel in (:stieltjes, :hilbert)
    @eval begin
        function $stiel(wP::Weighted, z::Number)
            P = wP.P
            w = orthogonalityweight(P)
            A,B,C = recurrencecoefficients(P)
            r1 = $stiel(w, z)*_p0(P) # stieltjes of the weight
            # (a[1]-z)*r[1] + b[1]r[2] == -sum(w)*_p0(P)
            # (a[1]/b[1]-z/b[1])*r[1] + r[2] == -sum(w)*_p0(P)/b[1]
            # (A[1]z + B[1])*r[1] - r[2] == A[1]sum(w)*_p0(P)
            # (A[1]z + B[1])*r[1]-A[1]sum(w)*_p0(P) ==  r[2] 
            r2 = (A[1]z + B[1])*r1-A[1]_stielsum($stiel, w)*_p0(P)
            transpose(RecurrenceArray(z, (A,B,C), [r1,r2]))
        end

        function $stiel(wP::Weighted, z::AbstractVector)
            P = wP.P
            A,B,C = recurrencecoefficients(P)
            w = orthogonalityweight(P)
            data1 = $stiel(w, z)
            μ = _p0(P)
            T = promote_type(eltype(data1), typeof(μ))
            data = Matrix{T}(undef, 2, length(z))
            data[1,:] .= data1 .* μ
            data[2,:] .= (A[1] .* z .+ B[1]) .* data[1,:] .- (A[1]_stielsum($stiel, w)*_p0(P))
            transpose(RecurrenceArray(z, (A,B,C), data))
        end
    end
end

sqrtx2(z::Number) = sqrt(z-1)*sqrt(z+1)
sqrtx2(x::Real) = sign(x)*sqrt(x^2-1)


stieltjes(P::Legendre, z...) = stieltjes(Weighted(P), z...)
stieltjes(J::AbstractJacobi{T}, z...) where T = stieltjes(Legendre{T}(), z...) * (Legendre{T}() \ J)
hilbert(P::Legendre, z...) = hilbert(Weighted(P), z...)
hilbert(J::AbstractJacobi{T}, z...) where T = hilbert(Legendre{T}(), z...) * (Legendre{T}() \ J)


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

const extrapolate = inbounds_getindex # TODO: bad pun?

for (stiel, stiel_lay, mapgetind) in ((:stieltjes, :stieltjes_layout, :extrapolate), (:hilbert, :hilbert_layout, :getindex))
    @eval begin
        function $stiel_lay(::MappedWeightLayout, w::SubQuasiArray{<:Any,1})
            m = parentindices(w)[1]
            # TODO: mapping other geometries
            P = parent(w)
            $stiel(P)[m]
        end

        function $stiel_lay(::MappedWeightLayout, w::AbstractQuasiVector, z::Number)
            m = basismap(w)
            # TODO: mapping other geometries
            P = demap(w)
            $stiel(P, $mapgetind(m, z))
        end

        $stiel_lay(::Union{MappedBasisLayouts, MappedOPLayouts}, wP::AbstractQuasiMatrix) = $stiel(demap(wP))[basismap(wP),:]

        function $stiel_lay(::Union{MappedBasisLayouts, MappedOPLayouts}, wP::AbstractQuasiMatrix, x::Inclusion)
            kr = basismap(wP)
            W = demap(wP)
            t̃ = axes(W,1)
            t = axes(wP,1)

            x == t && return $stiel(W)[kr,:]

            M = affine(t,t̃)
            @assert x isa Inclusion
            a,b = first(x),last(x)
            x̃ = Inclusion((M.A * a .+ M.b)..(M.A * b .+ M.b)) # map interval to new interval
            Q̃,M = arguments(*, $stiel(W, x̃))
            parent(Q̃)[affine(x,axes(parent(Q̃),1)),:] * M
        end

        function $stiel_lay(::Union{MappedBasisLayouts, MappedOPLayouts}, wT::AbstractQuasiMatrix, z::Number)
            P = demap(wT)
            m = basismap(wT)
            @assert m isa AbstractAffineQuasiVector
            z̃ = $mapgetind(m, z)
            $stiel(P, z̃)
        end

        function $stiel_lay(::Union{MappedBasisLayouts, MappedOPLayouts}, wT::AbstractQuasiMatrix, z::AbstractVector)
            P = demap(wT)
            m = basismap(wT)
            @assert m isa AbstractAffineQuasiVector
            z̃ = $mapgetind(m, z)
            $stiel(P, z̃)
        end
    end
end

###
# Interlace
###


function hilbert(W::PiecewiseInterlace{T}) where T
    Hs = [a == b ? hilbert(a) : stieltjes(b, axes(a,1))/convert(T,π) for a in W.args, b in W.args]
    N = length(W.args)
    Ts = [broadcastbasis(+, map(basis, Hs[k,:])...) for k=1:N]
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

function hilbert(S::PiecewiseInterlace, z::Number)
    @assert length(S.args) == 2
    a,b = S.args
    Sa = z in axes(a,1) ? hilbert(a, z) : stieltjes(a, z)/π
    Sb = z in axes(b,1) ? hilbert(b, z) : stieltjes(b, z)/π
    transpose(BlockBroadcastArray(vcat, unitblocks(transpose(Sa)), unitblocks(transpose(Sb))))
end

stieltjes(S::PiecewiseInterlace, z::AbstractVector) = Vcat((stieltjes(S, z) for z in z)...)