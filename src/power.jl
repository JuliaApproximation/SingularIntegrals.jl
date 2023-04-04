const PowerKernel{T,D1,D2,F<:Real} = BroadcastQuasiMatrix{T,typeof(^),Tuple{BroadcastQuasiMatrix{T,typeof(abs),Tuple{ConvKernel{T,Inclusion{T,D1},T,D2}}},F}}

# recognize structure of W = abs.(t .- x).^a
const PowerKernelPoint{T,W<:Number,V,D,A<:Number} = BroadcastQuasiMatrix{T,typeof(^),Tuple{BroadcastQuasiMatrix{T,typeof(abs),Tuple{ConvKernel{T,W,V,D}}},A}}


###
# PowerKernel
###

function *(K::PowerKernelPoint{<:Any,<:Number,<:Any,<:ChebyshevInterval,<:Number}, wC::UltrasphericalWeight)
    T = promote_type(eltype(K), eltype(wC))
    abscnv,α = K.args
    z,x = abscnv.args[1].args
    λ = wC.λ
    sqrt(convert(T,π))gamma(λ+one(T)/2)abs(z)^α*_₂F₁((1-α)/2, -α/2, 1+λ, 1/z^2)/gamma(1+λ)
end

*(K::PowerKernelPoint{<:Any,<:Number,<:Any,<:ChebyshevInterval,<:Number}, wC::LegendreWeight) = K * UltrasphericalWeight(wC)

@simplify function *(K::PowerKernelPoint{<:Any,<:Number,<:Any,<:ChebyshevInterval,<:Number}, wC::Weighted)
    T = promote_type(eltype(K), eltype(wC))
    cnv,α = K.args[1].args
    z,x = cnv.args
    λ = wC.P.λ

    sqrt(convert(T,π))gamma(λ+one(T)/2)abs(x)^α*_₂F₁((1-α)/2, -α/2, 1+λ, 1/x^2)/gamma(1+λ)

    β = (-α-1)/2
    error("Not implemented")
    # ChebyshevT{T}() * Diagonal(1:∞)
end

