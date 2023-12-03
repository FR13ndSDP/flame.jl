# #Range: 3 -> N+2*NG-2
function viscousFlux!(Fv_x, Fv_y, Q, NG, Nx, Ny, Pr, Cp, C_s, T_s, dξdx, dξdy, dηdx, dηdy, J)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y

    if i > Nx+2*NG-2 || j > Ny+2*NG-2 || i < 3 || j < 3
        return
    end

    @inbounds ∂ξ∂x = dξdx[i, j]
    @inbounds ∂ξ∂y = dξdy[i, j]
    @inbounds ∂η∂x = dηdx[i, j]
    @inbounds ∂η∂y = dηdy[i, j]

    @inbounds Jac = J[i, j]
    @inbounds T = Q[i, j, 5]
    μ = C_s * T * CUDA.sqrt(T)/(T + T_s)
    k = Cp*μ/Pr
    @inbounds ∂u∂ξ = 1/12*(Q[i-2, j, 2] - Q[i+2, j, 2]) - 2/3*(Q[i-1, j, 2] - Q[i+1, j, 2])
    @inbounds ∂u∂η = 1/12*(Q[i, j-2, 2] - Q[i, j+2, 2]) - 2/3*(Q[i, j-1, 2] - Q[i, j+1, 2])
    @inbounds ∂v∂ξ = 1/12*(Q[i-2, j, 3] - Q[i+2, j, 3]) - 2/3*(Q[i-1, j, 3] - Q[i+1, j, 3])
    @inbounds ∂v∂η = 1/12*(Q[i, j-2, 3] - Q[i, j+2, 3]) - 2/3*(Q[i, j-1, 3] - Q[i, j+1, 3])
    @inbounds ∂T∂ξ = 1/12*(Q[i-2, j, 5] - Q[i+2, j, 5]) - 2/3*(Q[i-1, j, 5] - Q[i+1, j, 5])
    @inbounds ∂T∂η = 1/12*(Q[i, j-2, 5] - Q[i, j+2, 5]) - 2/3*(Q[i, j-1, 5] - Q[i, j+1, 5])
    @inbounds u = Q[i, j, 2]
    @inbounds v = Q[i, j, 3]

    dudx = (∂u∂ξ * ∂ξ∂x + ∂u∂η * ∂η∂x) * Jac
    dudy = (∂u∂ξ * ∂ξ∂y + ∂u∂η * ∂η∂y) * Jac
    dvdx = (∂v∂ξ * ∂ξ∂x + ∂v∂η * ∂η∂x) * Jac
    dvdy = (∂v∂ξ * ∂ξ∂y + ∂v∂η * ∂η∂y) * Jac
    dTdx = (∂T∂ξ * ∂ξ∂x + ∂T∂η * ∂η∂x) * Jac
    dTdy = (∂T∂ξ * ∂ξ∂y + ∂T∂η * ∂η∂y) * Jac

    τ11 = μ*(4/3*dudx - 2/3*dvdy)
    τ12 = μ*(dudy + dvdx)
    τ22 = μ*(4/3*dvdy - 2/3*dudx)
    E1 = u * τ11 + v * τ12 + k * dTdx
    E2 = u * τ12 + v * τ22 + k * dTdy

    @inbounds Fv_x[i-2, j-2, 1] = ∂ξ∂x * τ11 + ∂ξ∂y * τ12
    @inbounds Fv_x[i-2, j-2, 2] = ∂ξ∂x * τ12 + ∂ξ∂y * τ22
    @inbounds Fv_x[i-2, j-2, 3] = ∂ξ∂x * E1 + ∂ξ∂y * E2

    @inbounds Fv_y[i-2, j-2, 1] = ∂η∂x * τ11 + ∂η∂y * τ12
    @inbounds Fv_y[i-2, j-2, 2] = ∂η∂x * τ12 + ∂η∂y * τ22
    @inbounds Fv_y[i-2, j-2, 3] = ∂η∂x * E1 + ∂η∂y * E2
    return
end
