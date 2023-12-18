include("reaction.jl")
using StaticArrays

# Range: 1 -> N+2*NG
function c2Prim(U, Q, Nx, Ny, NG, gamma, Rg)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    if i > Nx+2*NG || j > Ny+2*NG
        return
    end

    @inbounds ρ = U[i, j, 1] # ρ
    @inbounds u = U[i, j, 2]/ρ # U
    @inbounds v = U[i, j, 3]/ρ # V
    @inbounds p = CUDA.max((gamma-1) * (U[i, j, 4] - 0.5*ρ*(u*u + v*v)), CUDA.eps(Float64)) # P
    @inbounds T = CUDA.max(p/(Rg * ρ), CUDA.eps(Float64)) # T
    @inbounds c = CUDA.sqrt(gamma * Rg * T) # speed of sound

    @inbounds Q[i, j, 1] = ρ
    @inbounds Q[i, j, 2] = u
    @inbounds Q[i, j, 3] = v
    @inbounds Q[i, j, 4] = p
    @inbounds Q[i, j, 5] = T
    @inbounds Q[i, j, 6] = c
    return
end

function copyOld(Un, U, Nx, Ny, NG, NV)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y

    if i > Nx+2*NG || j > Ny+2*NG
        return
    end
    for n = 1:NV
        @inbounds Un[i, j, n] = U[i, j, n]
    end
    return
end

function linComb(U, Un, Nx, Ny, NG, NV, a::Float64, b::Float64)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y

    if i > Nx+2*NG || j > Ny+2*NG
        return
    end
    for n = 1:NV
        @inbounds U[i, j, n] = U[i, j, n] * a + Un[i, j, n] * b
    end
    return
end
