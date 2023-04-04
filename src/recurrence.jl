"""
    r = RecurrenceArray(z, (A, B, C), data)

is a vector corresponding to the non-domainant solution to the recurrence relationship, for `k = size(data,1)`

r[1:k,:] == data
r[k+1,j] == (A[k]z[j] + B[k])r[k,j] - C[k]*r[k-1,j]
"""
mutable struct RecurrenceArray{T, N, ZZ, AA<:AbstractVector, BB<:AbstractVector, CC<:AbstractVector} <: AbstractCachedArray{T,N}
    z::ZZ
    A::AA
    B::BB
    C::CC
    data::Array{T,N}
    datasize::NTuple{N,Int}
    p0::Vector{T} # stores p_{s-1} to determine when to switch to backward
    p1::Vector{T} # stores p_{s} to determine when to switch to backward
    u::Vector{T} # used for backsubstitution to store diagonal of U in LU
end

const RecurrenceVector{T, A<:AbstractVector, B<:AbstractVector, C<:AbstractVector} = RecurrenceArray{T, 1, T, A, B, C}
const RecurrenceMatrix{T, Z<:AbstractVector, A<:AbstractVector, B<:AbstractVector, C<:AbstractVector} = RecurrenceArray{T, 2, Z, A, B, C}

function RecurrenceArray(z::Number, (A,B,C), data::AbstractVector{T}) where T
    N = length(data)
    p0, p1 = initiateforwardrecurrence(N, A, B, C, z, one(z))
    RecurrenceVector{T,typeof(A),typeof(B),typeof(C)}(z, A, B, C, data, size(data), T[p0], T[p1], T[])
end

function RecurrenceArray(z::AbstractVector, (A,B,C), data::AbstractMatrix{T}) where T
    M,N = size(data)
    p0 = Vector{T}(undef, N)
    p1 = Vector{T}(undef, N)
    for j = 1:length(z)
        p0[j], p1[j] = initiateforwardrecurrence(N, A, B, C, z[j], one(z))
    end
    RecurrenceMatrix{T,typeof(z),typeof(A),typeof(B),typeof(C)}(z, A, B, C, data, size(data), p0, p1, T[])
end

size(R::RecurrenceVector) = (ℵ₀,) # potential to add maximum size of operator
size(R::RecurrenceMatrix) = (ℵ₀, size(R.data,2)) # potential to add maximum size of operator
copy(R::RecurrenceArray) = R # immutable entries

# to estimate error in forward recurrence we compute the dominant solution (the OPs) simeultaneously

function resizedata!(K::RecurrenceVector, n)
    n ≤ 0 && return K
    @boundscheck checkbounds(Bool, K, n) || throw(ArgumentError("Cannot resize beyond size of operator"))

    # increase size of array if necessary
    olddata = cacheddata(K)
    ν, = K.datasize
    n = max(ν,n)
    if n > length(K.data) # double memory to avoid O(n^2) growing
        K.data = similar(K.data, min(2n,length(K)))
        K.data[axes(olddata,1)] = olddata
    end
    if n > ν
        A,B,C = K.A,K.B,K.C
        z = K.z
        tol = 100
        if ν > 2 && iszero(K.data[ν-1]) && iszero(K.data[ν])
            # no data
            zero!(view(K.data, ν+1:n))
        else
            p0, p1 = K.p0[1], K.p1[1]
            k = ν
            while abs(p1) < tol*k && k < n
                p1,p0 = _forwardrecurrence_next(k, A, B, C, z, p0, p1),p1
                k += 1
            end
            K.p0[1], K.p1[1] = p0, p1
            if k > ν
                _forwardrecurrence!(K.data, A, B, C, z, ν:k)
            end
            if k < n
                backwardrecurrence!(K, A, B, C, z, k, n)
            end
        end

        K.datasize = (max(K.datasize[1],n),)
    end
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