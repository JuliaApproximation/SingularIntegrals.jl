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