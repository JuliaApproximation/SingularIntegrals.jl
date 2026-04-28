########
# Inspired by
# https://github.com/marcusdavidwebb/MTFun.jl
########

using SingularIntegrals, ClassicalOrthogonalPolynomials

φ = (n,x) -> sqrt(1/π) * (im - x)^float(n) / (im + x)^(float(n)+1)
R = (n,z) -> ((z-im)/(z+im))^n - 1
p = n -> expand(legendre(-100..100), x -> φ(n,x))

@test [p(k)'p(j) for k=-4:4, j=-4:4] ≈ I rtol=2E-2


for n = 0:3
    z = 3+2im
    @test cauchy(p(n), z) ≈ φ(n,z) rtol=5E-2
    z = 3 - 2im
    @test cauchy(p(n), z) ≈ 0 atol=5E-2

    @test sqrt(π)*(-1)^n*φ(n,0.1) ≈ -im*(R(n,0.1) - R(n+1,0.1))/2
end

for n = -3:-1
    z = 3+2im
    @test cauchy(p(n), z) ≈ 0 atol=5E-2
    z = 3 - 2im
    @test cauchy(p(n), z) ≈ -φ(n,z) rtol=5E-2

    @test sqrt(π)*(-1)^n*φ(n,0.1) ≈ -im*(R(n,0.1) - R(n+1,0.1))/2
end


x = 0.1
for n = 0:5
    @test (-1)^n * sqrt(2/π) * (1+2im*x)^n / (1-2im*x)^(n+1) ≈ sum(expand(chebyshevt(0..100), k -> exp(-k/2) * laguerrel(n,k) * exp(im*k*x)))/sqrt(2π)
end

for n  = -5:-1
    @test (-1)^n * sqrt(2/π) * (1+2im*x)^n / (1-2im*x)^(n+1) ≈ -sum(expand(chebyshevt(0..100), k -> exp(-k/2) * laguerrel(abs(n)-1,k) * exp(-im*k*x)))/sqrt(2π) 
end

 φ = (n,x) -> (-1)^n * sqrt(2/π) * (1+2im*x)^n / (1-2im*x)^(n+1)

# ∫ exp(-k/2) * L(n,k) * exp(i*k*z)) dx =
# ∫ exp(-k) * L(n,k) * exp(k*(i*z+1/2)) dx = 
# ∫ d/dk(k * exp(-k) * L(n-1,1,k)) * exp(k*(i*z+1/2)) dx/n = 
# -∫ k * exp(-k) * L(n-1,1,k) *d/dk(exp(k*(i*z+1/2))) dx/n = 
# -∫ k * exp(-k) * L(n-1,1,k) *exp(k*(i*z+1/2)) dx * (i*z+1/2)/n = 
# -∫ exp(-k) * (-L(n,k)+L(n-1,k))   *exp(k*(i*z+1/2)) dx * (i*z+1/2) = 

n = 2
@test φ(n,x) ≈ (φ(n,x)-φ(n-1,x)) * (im*x+1/2)
@test φ(n,x)≈ φ(n-1,x) * (im*x+1/2)/(im*x-1/2)

# F^{-1}[φ] = 1/sqrt(2π) * ∫ φ(x) exp(-im*k*z) dx =  exp(-k/2) * L(n,k)
# F^{-1}[exp(iωx)φ] =  1/sqrt(2π) * ∫ φ(x) exp(i(ω-k)x) dx = exp(-(k-ω)/2) * L(n,k-ω)


# ∫_ω sign(k) exp(-k/2) * L(n,k) * exp(im*k*x) dx =
# ∫_ω sign(k) exp(-k) * L(n,k) * exp(k*(im*x+1/2)) dx = 
# ∫_ω d/dk(k * exp(-k) * L(n-1,1,k)) * exp(k*(im*x+1/2)) dx/n = 
# -ω*exp(-ω) -∫ k * exp(-k) * L(n-1,1,k) *d/dk(exp(k*(im*z+1/2))) dx/n = 