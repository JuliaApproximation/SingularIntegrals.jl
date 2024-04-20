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

RecurrenceArray(z, A, B, C, data::Array{T,N}, datasize, p0, p1) where {T,N} = RecurrenceArray{T,N,typeof(z),typeof(A),typeof(B),typeof(C)}(z, A, B, C, data, datasize, p0, p1, T[])

function RecurrenceArray(z::Number, (A,B,C), data::AbstractVector{T}) where T
    N = length(data)
    p0, p1 = initiateforwardrecurrence(N, A, B, C, z, one(z))
    if iszero(p1)
        p1 = one(p1) # avoid degeneracy in recurrence. Probably needs more thought
    end
    RecurrenceVector{T,typeof(A),typeof(B),typeof(C)}(z, A, B, C, data, size(data), T[p0], T[p1], T[])
end

function RecurrenceArray(z::AbstractVector, (A,B,C), data::AbstractMatrix{T}) where T
    M,N = size(data)
    p0 = Vector{T}(undef, N)
    p1 = Vector{T}(undef, N)
    for j = axes(z,1)
        p0[j], p1[j] = initiateforwardrecurrence(M, A, B, C, z[j], one(T))
        if iszero(p1[j])
            p1[j] = one(p1[j]) # avoid degeneracy in recurrence. Probably needs more thought
        end
    end
    RecurrenceMatrix{T,typeof(z),typeof(A),typeof(B),typeof(C)}(z, A, B, C, data, size(data), p0, p1, T[])
end

size(R::RecurrenceVector) = (ℵ₀,) # potential to add maximum size of operator
size(R::RecurrenceMatrix) = (ℵ₀, size(R.data,2)) # potential to add maximum size of operator
copy(R::RecurrenceArray) = R # immutable entries



function _growdata!(B::AbstractArray{<:Any,N}, nm::Vararg{Integer,N}) where N
    # increase size of array if necessary
    olddata = B.data
    νμ = size(olddata)
    nm = max.(νμ,nm)
    if νμ ≠ nm
        B.data = similar(B.data, nm...)
        B.data[axes(olddata)...] = olddata
    end
end


# to estimate error in forward recurrence we compute the dominant solution (the OPs) simeultaneously
function resizedata!(K::RecurrenceArray, m, n...)
    m ≤ 0 && return K

    # increase size of array if necessary
    _growdata!(K, m, n...)
    ν = K.datasize[1]
    if m > ν
        A,B,C = K.A,K.B,K.C
        tol = 100
        for j = axes(K.z,1)
            z = K.z[j]
        
            if ν > 2 && iszero(K.data[ν-1,j]) && iszero(K.data[ν,j])
                # no data
                zero!(view(K.data, ν+1:m, j))
            else
                p0, p1 = K.p0[j], K.p1[j]
                k = ν
                while abs(p1) < tol*k && k < m
                    p1,p0 = _forwardrecurrence_next(k, A, B, C, z, p0, p1),p1
                    k += 1
                end
                K.p0[j], K.p1[j] = p0, p1
                if k > ν
                    _forwardrecurrence!(view(K.data,:,j), A, B, C, z, ν:k)
                end
                if k < m
                    if K isa AbstractVector
                        backwardrecurrence!(K, A, B, C, z, k:m)
                    else
                        backwardrecurrence!(K, A, B, C, z, k:m, j)
                    end
                end
            end
        end
        K.datasize = (max(K.datasize[1],m), tail(K.datasize)...)
    end
end


function backwardrecurrence!(K, A, B, C, z, nN::AbstractUnitRange, j...)
    n,N = first(nN),last(nN)
    T = eltype(z)
    tol = 100eps(real(T))
    maxiterations = 100_000_000
    data = K.data
    u = K.u
    resize!(u, max(length(u), N))
    # we use data as a working vector and do an inplace LU
    # r[n+1] - (A[n]z + B[n])r[n] + C[n] r[n-1] == 0
    u[n+1] = -(A[n+1]z + B[n+1])
    data[n+1, j...] = -C[n+1]*data[n, j...]

    # forward elimination
    k = n+1
    while abs(data[k,j...]) > tol
        k ≥ maxiterations && error("maximum iterations reached")
        if k == N
            # need to resize data, lets use rate of decay as estimate
            μ = min(abs(data[k,j...]/data[k-1,j...]), abs(data[k-1,j...]/data[k-2,j...]))
            # data[k] * μ^M ≤ ε
            #  M ≥ log(ε/data[k])/log(μ)
            N = ceil(Int, max(2N, min(maxiterations, log(eps(real(T))/100)/log(μ+eps(real(T))))))
            _growdata!(K, N, j...)
            resize!(u, N)
            data = K.data
        end
        ℓ = -C[k+1]/u[k]
        u[k+1] = ℓ-(A[k+1]z + B[k+1])
        data[k+1,j...] = ℓ*data[k,j...]
        k += 1
    end

    data[k,j...] /= u[k]

    # back-sub
    for κ = k-1:-1:n+1
        data[κ,j...] = (data[κ,j...] - data[κ+1,j...])/u[κ]
    end
    
    for κ = k+1:N
        data[κ,j...] = 0
    end
    K
end


###
# override indexing to resize first
####

function _getindex_resize_iffinite!(A, kr, jr, m::Int)
    resizedata!(A, m, size(A,2))
    A.data[kr,jr]
end

_getindex_resize_iffinite!(A, kr, jr, _) = layout_getindex(A, kr, jr)

@inline getindex(A::RecurrenceMatrix, kr::AbstractUnitRange, jr::AbstractUnitRange) = _getindex_resize_iffinite!(A, kr, jr, maximum(kr))
@inline getindex(A::RecurrenceMatrix, kr::AbstractVector, jr::AbstractVector) = _getindex_resize_iffinite!(A, kr, jr, maximum(kr))
@inline getindex(A::RecurrenceMatrix, k::Integer, jr::AbstractVector) = _getindex_resize_iffinite!(A, k, jr, k)
@inline getindex(A::RecurrenceMatrix, k::Integer, jr::AbstractUnitRange) = _getindex_resize_iffinite!(A, k, jr, k)
@inline getindex(A::RecurrenceMatrix, k::Integer, ::Colon) = _getindex_resize_iffinite!(A, k, :, k)
@inline getindex(A::RecurrenceMatrix, kr::AbstractVector, ::Colon) = _getindex_resize_iffinite!(A, kr, :, maximum(kr))
@inline getindex(A::RecurrenceMatrix, kr::AbstractUnitRange, ::Colon) = _getindex_resize_iffinite!(A, kr, :, maximum(kr))

function view(A::RecurrenceVector, kr::AbstractVector)
    resizedata!(A, maximum(kr))
    view(A.data, kr)
end

###
# broadcasted
###
broadcasted(::LazyArrayStyle, op, A::Transpose{<:Any,<:RecurrenceArray}) = transpose(op.(parent(A)))

broadcasted(::LazyArrayStyle, ::typeof(*), c::Number, A::RecurrenceArray) = RecurrenceArray(A.z, A.A, A.B, A.C, c .* A.data, A.datasize, c .* A.p0, c .* A.p1)
function recurrence_broadcasted(op, A::RecurrenceMatrix, x::AbstractVector)
    p = paddeddata(x)
    n = size(p,1)
    resizedata!(A, n, size(p,2))
    data = copy(A.data)
    data[1:n,:] .+= p
    RecurrenceArray(A.z, A.A, A.B, A.C, data, A.datasize, A.p0, A.p1)
end

function recurrence_broadcasted(op, A::RecurrenceVector, x::AbstractVector)
    p = paddeddata(x)
    n = size(p,1)
    resizedata!(A, n)
    data = copy(A.data)
    data[1:n] .+= p
    RecurrenceArray(A.z, A.A, A.B, A.C, data, A.datasize, A.p0, A.p1)
end

for op in (:+, :-)
    @eval begin
        broadcasted(::LazyArrayStyle, ::typeof($op), A::RecurrenceArray, x::AbstractVector) = recurrence_broadcasted($op, A, x)
        broadcasted(::LazyArrayStyle, ::typeof($op), A::RecurrenceVector, x::Vcat{<:Any,1}) = recurrence_broadcasted($op, A, x)
    end
end