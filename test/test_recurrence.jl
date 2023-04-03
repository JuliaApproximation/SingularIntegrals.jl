using SingularIntegrals, ClassicalOrthogonalPolynomials
using SingularIntegrals: RecurrenceArray
using ClassicalOrthogonalPolynomials: recurrencecoefficients


@testset "RecurrenceArray" begin
    T,U = ChebyshevT(),ChebyshevU()
    r = RecurrenceArray(0.1, recurrencecoefficients(T), [0.0, 1.0])
    @test r[1:1000] ≈ [0; U[0.1,1:999]]
    @test r[10_000] ≈ U[0.1,9_999]

    r = RecurrenceArray(1.0, recurrencecoefficients(T), [0.0, 1.0])
    @test r[1:1000] ≈ [0.0; U[1,1:999]]
    @test r[10_000] ≈ U[1.0,9_999]

    for z in (1.000000001, 1.000001, -1.000001, 10.0, -10.0)
        ξ = inv(z + sign(z)sqrt(z^2-1))
        r = RecurrenceArray(z, recurrencecoefficients(U), [ξ,ξ^2])
        @test r[1:1000] ≈ ξ.^(1:1000)
        @test r[10_000] ≈ ξ.^(10_000)
    end
end
