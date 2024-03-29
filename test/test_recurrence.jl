using SingularIntegrals, ClassicalOrthogonalPolynomials, Test
using SingularIntegrals: RecurrenceArray
using ClassicalOrthogonalPolynomials: recurrencecoefficients


@testset "RecurrenceArray" begin
    @testset "RecurrenceVector" begin
        T,U = ChebyshevT(),ChebyshevU()
        for z in (0.1, 1.0)
            r = RecurrenceArray(z, recurrencecoefficients(T), [0.0, 1.0])
            @test r[1:1000] ≈ [0; U[z,1:999]]
            @test r[10_000] ≈ U[z,9_999]

            r = RecurrenceArray(z, recurrencecoefficients(U), [1.0, z])
            @test r[1:1000] ≈ T[z,1:1000]
            @test r[10_000] ≈ T[z,10_000]
        end

        for z in (1.000000001, 1.000001, -1.000001, 10.0, -10.0)
            ξ = inv(z + sign(z)sqrt(z^2-1))
            r = RecurrenceArray(z, recurrencecoefficients(U), [ξ,ξ^2])
            @test r[1:1000] ≈ ξ.^(1:1000)
            @test r[10_000] ≈ ξ.^(10_000) atol=3E-10
        end

        for z in (0.2567881003580743 - 0.33437737333561895im)
            ξ = π * (z - √(z-1) * √(z+1))
            r = RecurrenceArray(z, recurrencecoefficients(U), [ξ,2z*ξ-π])
            @test sizeof(r.data) < 1000_000
        end
    end

    @testset "RecurrenceMatrix" begin
        U = ChebyshevU()
        z = [2.,3.,100.]
        ξ = @. inv(z + sign(z)sqrt(z^2-1))
        r = RecurrenceArray(z, recurrencecoefficients(U), [ξ'; ξ'.^2])
        @test r[1:100,:] ≈ [RecurrenceArray(z[j], recurrencecoefficients(U),  [ξ[j],ξ[j]^2])[k] for k=1:100, j=axes(z,1)]
        @test r[1:100,:] ≈ r[1:100,1:3] ≈ r[collect(1:100),1:3] ≈ r[1:100,collect(1:3)] ≈ r[collect(1:100),:]
        @test r[100,:] ≈ r[100,1:3]
    end
end
