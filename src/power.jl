const PowerKernel{T,D1,D2,F<:Real} = BroadcastQuasiMatrix{T,typeof(^),Tuple{BroadcastQuasiMatrix{T,typeof(abs),Tuple{ConvKernel{T,Inclusion{T,D1},T,D2}}},F}}

# recognize structure of W = abs.(t .- x).^a
const PowerKernelPoint{T,W<:Number,V,D,A<:Number} = BroadcastQuasiMatrix{T,typeof(^),Tuple{BroadcastQuasiMatrix{T,typeof(abs),Tuple{ConvKernel{T,W,V,D}}},A}}


###
# PowerKernel
###


function powerlawmoment(::Val{0}, α, λ, z::Real)
    T = promote_type(typeof(α), typeof(λ), typeof(z))
    if -1 ≤ z ≤ 1
        beta((α+one(T))/2, λ+one(T)/2)_₂F₁(-α/2, -λ-α/2, one(T)/2, z^2)
    else
        beta(one(T)/2, λ+one(T)/2)abs(z)^α*_₂F₁((1-α)/2, -α/2, 1+λ, 1/z^2)
    end
end

function powerlawmoment(::Val{1}, α, λ, z::Real)
    T = promote_type(typeof(α), typeof(λ), typeof(z))
    if -1 ≤ z ≤ 1
        -2^(2-α)*λ*sqrt(convert(T,π))*gamma(α+2)*gamma(λ+one(T)/2)/(gamma(α/2)*gamma(α/2+λ+1))*((2*z^2*(α+λ)+1)*_₂F₁(-α/2,-λ-α/2,one(T)/2,z^2)+(z^2-1)*_₂F₁(-α/2,-λ-α/2,-one(T)/2,z^2))/(α*(α+1)*(α+2*λ+1)*z)+2*λ*z*gamma((α+1)/2)*gamma(λ+one(T)/2)/(gamma(λ+α/2+1))*_₂F₁(-α/2,-λ-α/2,one(T)/2,z^2)
    else
        -sign(z)α*λ*beta(one(T)/2, λ+one(T)/2)*abs(z)^(α-1)*_₂F₁((1-α)/2, 1-α/2, 2+λ, 1/z^2)/(1+λ)
    end
end


function *(K::PowerKernelPoint{<:Any,<:Number,<:Any,<:ChebyshevInterval,<:Number}, wC::UltrasphericalWeight)
    abscnv,α = K.args
    z,x = abscnv.args[1].args
    powerlawmoment(Val(0), α, wC.λ, z)
end

*(K::PowerKernelPoint{<:Any,<:Number,<:Any,<:ChebyshevInterval,<:Number}, wC::LegendreWeight) = K * UltrasphericalWeight(wC)

function powerlawrecurrence(α, λ)
    T = promote_type(typeof(α), typeof(λ))
    n = 0:∞
    A = @. 2*(λ + n) * (2λ + n) / ((n+1)*(2λ+n+α+1))
    B = Zeros{T}(∞)
    C = @. (n-α-1)*(2λ+n-1)*(2λ+n)/(n*(n+1)*(2λ+n+α+1))
    A,B,C
end

@simplify function *(K::PowerKernelPoint{<:Any,<:Number,<:Any,<:ChebyshevInterval,<:Number}, wC::Weighted)
    T = promote_type(eltype(K), eltype(wC))
    cnv,α = K.args
    z,x = cnv.args[1].args
    λ = wC.P.λ
    transpose(RecurrenceArray(z, powerlawrecurrence(α, λ), [powerlawmoment(Val(0), α, λ, z), powerlawmoment(Val(1), α, λ, z)]))
end

@simplify function *(K::PowerKernelPoint{<:Any,<:Number,<:Any,<:ChebyshevInterval,<:Number}, wC::Legendre)
    T = eltype(wC)
    K * Weighted(Ultraspherical{T}(one(T)/2))
end

