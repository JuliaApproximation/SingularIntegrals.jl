"""
    r = RecurrenceArray(z, (A, B, C), data)

is a vector corresponding to the non-domainant solution to the recurrence relationship, for `k = size(data,1)`

r[1:k,:] == data
r[k+1,j] == (A[k]z[j] + B[k])r[k,j] - C[k]*r[k-1,j]
"""
mutable struct RecurrenceArray{T, N, AA<:AbstractVector, BB<:AbstractVector, CC<:AbstractVector} <: AbstractCachedArray{T,N}
    z::T
    A::AA
    B::BB
    C::CC
    data::Array{T,N}
    datasize::NTuple{N,Int}
    p0::Vector{T} # stores p_{s-1} to determine when to switch to backward
    p1::Vector{T} # stores p_{s} to determine when to switch to backward
    u::Vector{T} # used for backsubstitution to store diagonal of U in LU
end

const RecurrenceVector{T, A<:AbstractVector, B<:AbstractVector, C<:AbstractVector} = RecurrenceArray{T, 1, A, B, C}
const RecurrenceMatrix{T, A<:AbstractVector, B<:AbstractVector, C<:AbstractVector} = RecurrenceArray{T, 2, A, B, C}

function RecurrenceArray(z::Number, (A,B,C), data::AbstractVector{T}) where T
    N = length(data)
    p0, p1 = initiateforwardrecurrence(N, A, B, C, z, one(z))
    RecurrenceVector{T,typeof(A),typeof(B),typeof(C)}(z, A, B, C, data, (length(data),), T[p0], T[p1], T[])
end

size(R::RecurrenceVector) = (ℵ₀,) # potential to add maximum size of operator
size(R::RecurrenceMatrix) = (ℵ₀, size(R.data,2)) # potential to add maximum size of operator
copy(R::RecurrenceArray) = R # immutable entries

# to estimate error in forward recurrence we compute the dominant solution (the OPs) simeultaneously

function cache_filldata!(K::RecurrenceVector, kr)
    s = K.datasize[1]
    A,B,C = K.A,K.B,K.C
    z = K.z
    N = maximum(kr)
    tol = 100N
    if s > 2 && iszero(K.data[s-1]) && iszero(K.data[s])
        # no data
        zero!(view(K.data, s+1:N))
    else
        p0, p1 = K.p0[1], K.p1[1]
        n = s
        while abs(p1) < tol && n < N
            p1,p0 = _forwardrecurrence_next(n, A, B, C, z, p0, p1),p1
            n += 1
        end
        K.p0[1], K.p1[1] = p0, p1
        if n > s
            __forwardrecurrence!(K.data, A, B, C, z, s, n)
        end
        if n < N
            backwardrecurrence!(K, A, B, C, z, n, N)
        end
    end

    K.datasize = (max(K.datasize[1],N),)
end

function backwardrecurrence!(K, A, B, C, z, n, N)
    T = eltype(z)
    tol = 1E-14
    maxiterations = 100_000_000
    data = K.data
    u = K.u
    resize!(u, max(length(u), N))
    # we use data as a working vector and do an inplace LU
    # r[n+1] - (A[n]z + B[n])r[n] + C[n] r[n-1] == 0
    u[n+1] = -(A[n+1]z + B[n+1])
    data[n+1] = -C[n+1]*data[n]

    # forward elimination
    k = n+1
    while abs(data[k]) > tol
        k ≥ maxiterations && error("maximum iterations reached")
        if k == N
            # need to resize data, lets use rate of decay as estimate
            μ = min(abs(data[k]/data[k-1]), abs(data[k-1]/data[k-2]))
            # data[k] * μ^M ≤ ε
            #  M ≥ log(ε/data[k])/log(μ)
            N = ceil(Int, max(2N, min(maxiterations, log(eps(real(T))/100)/log(μ))))
            resize!(data, N)
            resize!(u, N)
        end
        ℓ = -C[k+1]/u[k]
        u[k+1] = ℓ-(A[k+1]z + B[k+1])
        data[k+1] = ℓ*data[k]
        k += 1
    end

    data[k] /= u[k]

    # back-sub
    for κ = k-1:-1:n+1
        data[κ] = (data[κ] - data[κ+1])/u[κ]
    end
    
    for κ = k+1:N
        data[κ] = 0
    end
    K.datasize = (max(K.datasize[1],N),)
    K
end