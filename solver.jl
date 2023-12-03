using WriteVTK
using CUDA
using JLD2
CUDA.allowscalar(false)

@inline function minmod(a, b)
    ifelse(a*b > 0, (CUDA.abs(a) > CUDA.abs(b)) ? b : a, 0)
end

#Range: 1 -> N-1
function NND_x!(F, Fp, Fm, NG, Nx, Ny)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y

    if i > Nx-1 || j > Ny-2
        return
    end
    for n = 1:4
        @inbounds fp = Fp[i+NG, j+1+NG, n] + 0.5*minmod(Fp[i+1+NG, j+1+NG, n]-Fp[i+NG, j+1+NG, n], Fp[i+NG, j+1+NG, n] - Fp[i-1+NG, j+1+NG, n])
        @inbounds fm = Fm[i+1+NG, j+1+NG, n] - 0.5*minmod(Fm[i+2+NG, j+1+NG, n]-Fm[i+1+NG, j+1+NG, n], Fm[i+1+NG, j+1+NG, n] - Fm[i+NG, j+1+NG, n])
        @inbounds F[i, j, n] = fp + fm
    end
    return
end

function NND_y!(F, Fp, Fm, NG, Nx, Ny)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y

    if i > Nx-2 || j > Ny-1
        return
    end
    for n = 1:4
        @inbounds fp = Fp[i+1+NG, j+NG, n] + 0.5*minmod(Fp[i+1+NG, j+1+NG, n]-Fp[i+1+NG, j+NG, n], Fp[i+1+NG, j+NG, n] - Fp[i+1+NG, j-1+NG, n])
        @inbounds fm = Fm[i+1+NG, j+1+NG, n] - 0.5*minmod(Fm[i+1+NG, j+2+NG, n]-Fm[i+1+NG, j+1+NG, n], Fm[i+1+NG, j+1+NG, n] - Fm[i+1+NG, j+NG, n])
        @inbounds F[i, j, n] = fp + fm
    end
    return
end

# Range: 1 -> N+2*NG
function c2Prim!(U, Q, Nx, Ny, NG, gamma, Rg)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    if i > Nx+2*NG || j > Ny+2*NG
        return
    end

    @inbounds Q[i, j, 1] = U[i, j, 1] # ρ
    @inbounds Q[i, j, 2] = U[i, j, 2]/U[i, j, 1] # U
    @inbounds Q[i, j, 3] = U[i, j, 3]/U[i, j, 1] # V
    @inbounds Q[i, j, 4] = (gamma-1) * (U[i, j, 4] - 0.5*Q[i, j, 1]*(Q[i, j, 2]^2 + Q[i, j, 3]^2)) # P
    @inbounds Q[i, j, 5] = Q[i, j, 4]/(Rg * Q[i, j, 1]) # T
    @inbounds Q[i, j, 6] = CUDA.sqrt(gamma * Q[i, j, 4] / Q[i, j, 1]) # speed of sound
    return
end

#Range: 1->N+2*NG
function fluxSplit!(Q, Fp, Fm, Nx, Ny, NG, Ax, Ay, gamma, tmp0, split_C1, split_C3)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    if i > Nx+2*NG || j > Ny+2*NG
        return
    end

    @inbounds ρ = Q[i, j, 1]
    @inbounds u = Q[i, j, 2]
    @inbounds v = Q[i, j, 3]
    @inbounds c = Q[i, j, 6]
    @inbounds A1 = Ax[i, j]
    @inbounds A2 = Ay[i, j]

    ss = CUDA.sqrt(A1^2 + A2^2)
    E1 = A1*u + A2*v
    E2 = E1 - c*ss
    E3 = E1 + c*ss

    ss = 1/ss

    A1 *= ss
    A2 *= ss

    E1P = (E1 + CUDA.sqrt(E1 * E1)) * 0.5
    E2P = (E2 + CUDA.sqrt(E2 * E2)) * 0.5
    E3P = (E3 + CUDA.sqrt(E3 * E3)) * 0.5

    E1M = E1 - E1P
    E2M = E2 - E2P
    E3M = E3 - E3P

    uc1 = u - c * A1
    uc2 = u + c * A1
    vc1 = v - c * A2
    vc2 = v + c * A2

    vvc1 = (uc1^2 + vc1^2) * 0.50
    vvc2 = (uc2^2 + vc2^2) * 0.50
    vv = (gamma - 1.0) * (u^2 + v^2)
    W2 = split_C3 * c^2

    tmp1 = ρ * tmp0
    @inbounds Fp[i, j, 1] = tmp1 * (split_C1 * E1P + E2P + E3P);
    @inbounds Fp[i, j, 2] = tmp1 * (split_C1 * E1P * u + E2P * uc1 + E3P * uc2)
    @inbounds Fp[i, j, 3] = tmp1 * (split_C1 * E1P * v + E2P * vc1 + E3P * vc2)
    @inbounds Fp[i, j, 4] = tmp1 * (E1P * vv + E2P * vvc1 + E3P * vvc2 + W2 * (E2P + E3P))

    @inbounds Fm[i, j, 1] = tmp1 * (split_C1 * E1M + E2M + E3M);
    @inbounds Fm[i, j, 2] = tmp1 * (split_C1 * E1M * u + E2M * uc1 + E3M * uc2);
    @inbounds Fm[i, j, 3] = tmp1 * (split_C1 * E1M * v + E2M * vc1 + E3M * vc2);
    @inbounds Fm[i, j, 4] = tmp1 * (E1M * vv + E2M * vvc1 + E3M * vvc2 + W2 * (E2M + E3M));
    return 
end

#Range: 2+NG -> N+NG-1
function div!(U, Fx, Fy, Fv_x, Fv_y, dt, NG, Nx, Ny, J)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    if i > Nx+NG-1 || i < 2+NG || j > Ny+NG-1 || j < 2+NG
        return
    end

    @inbounds Jac = J[i, j]
    @inbounds dV11dξ = -1/60*(Fv_x[i-1-NG,j+2-NG,1] - Fv_x[i+5-NG, j+2-NG, 1]) + 3/20 *(Fv_x[i-NG,j+2-NG, 1] - Fv_x[i+4-NG, j+2-NG, 1]) - 3/4*(Fv_x[i+1-NG,j+2-NG, 1] - Fv_x[i+3-NG, j+2-NG, 1])
    @inbounds dV12dξ = -1/60*(Fv_x[i-1-NG,j+2-NG,2] - Fv_x[i+5-NG, j+2-NG, 2]) + 3/20 *(Fv_x[i-NG,j+2-NG, 2] - Fv_x[i+4-NG, j+2-NG, 2]) - 3/4*(Fv_x[i+1-NG,j+2-NG, 2] - Fv_x[i+3-NG, j+2-NG, 2])
    @inbounds dV13dξ = -1/60*(Fv_x[i-1-NG,j+2-NG,3] - Fv_x[i+5-NG, j+2-NG, 3]) + 3/20 *(Fv_x[i-NG,j+2-NG, 3] - Fv_x[i+4-NG, j+2-NG, 3]) - 3/4*(Fv_x[i+1-NG,j+2-NG, 3] - Fv_x[i+3-NG, j+2-NG, 3])

    @inbounds dV21dη = -1/60*(Fv_y[i+2-NG,j-1-NG,1] - Fv_y[i+2-NG, j+5-NG, 1]) + 3/20 *(Fv_y[i+2-NG,j-NG, 1] - Fv_y[i+2-NG, j+4-NG, 1]) - 3/4*(Fv_y[i+2-NG,j+1-NG, 1] - Fv_y[i+2-NG, j+3-NG, 1])
    @inbounds dV22dη = -1/60*(Fv_y[i+2-NG,j-1-NG,2] - Fv_y[i+2-NG, j+5-NG, 2]) + 3/20 *(Fv_y[i+2-NG,j-NG, 2] - Fv_y[i+2-NG, j+4-NG, 2]) - 3/4*(Fv_y[i+2-NG,j+1-NG, 2] - Fv_y[i+2-NG, j+3-NG, 2])
    @inbounds dV23dη = -1/60*(Fv_y[i+2-NG,j-1-NG,3] - Fv_y[i+2-NG, j+5-NG, 3]) + 3/20 *(Fv_y[i+2-NG,j-NG, 3] - Fv_y[i+2-NG, j+4-NG, 3]) - 3/4*(Fv_y[i+2-NG,j+1-NG, 3] - Fv_y[i+2-NG, j+3-NG, 3])

    for n = 1:4
        @inbounds U[i, j, n] +=  (Fx[i-1-NG, j-1-NG, n] - Fx[i-NG, j-1-NG, n] + Fy[i-1-NG, j-1-NG, n] - Fy[i-1-NG, j-NG, n]) * dt * Jac
    end
    @inbounds U[i, j, 2] += (dV11dξ + dV21dη) * dt * Jac
    @inbounds U[i, j, 3] += (dV12dξ + dV22dη) * dt * Jac
    @inbounds U[i, j, 4] += (dV13dξ + dV23dη) * dt * Jac
    return
end

function fillGhost!(U, NG, Nx, Ny, gamma, T_wall, Rg)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    
    if i > Nx+2*NG || j > Ny+2*NG
        return
    end

    # Mach 6 inlet
    if i <= 10
        @inbounds U[i, j, 1] = 0.04468971904908923
        @inbounds U[i, j, 2] = 0.04468971904908923 * 1068.980448838986
        @inbounds U[i, j, 3] = 0.0
        @inbounds U[i, j, 4] = 0.01*101325/0.4 + 0.5*0.04468971904908923*1068.980448838986^2
    elseif i > Nx + NG -1
        for n = 1:4
            @inbounds U[i, j, n] = U[Nx+NG-1, j, n]
        end
    else
        if j == NG+1 
            @inbounds U[i, j, 2] = 0
            @inbounds U[i, j, 3] = 0
            @inbounds U[i, j, 4] = U[i, j+1, 4] - 0.5/U[i, j+1, 1]*(U[i, j+1, 2]^2 + U[i, j+1, 3]^2)
            @inbounds U[i, j, 1] = U[i, j, 4] * (gamma-1)/(T_wall * Rg)
        elseif j < NG+1
            p = (gamma-1) * (U[i, 2*NG+2-j, 4] - 0.5/U[i, 2*NG+2-j, 1]*(U[i, 2*NG+2-j, 2]^2 + U[i, 2*NG+2-j, 3]^2))
            @inbounds U[i, j, 1] = p/(Rg * T_wall)
            @inbounds U[i, j, 2] = -U[i, 2*NG+2-j, 2]/U[i, 2*NG+2-j, 1] * U[i, j, 1]
            @inbounds U[i, j, 3] = -U[i, 2*NG+2-j, 3]/U[i, 2*NG+2-j, 1] * U[i, j, 1]
            @inbounds U[i, j, 4] = p/(gamma-1) + 0.5/U[i, j, 1]*(U[i, j, 2]^2 + U[i, j, 3]^2)
        elseif j > Ny+NG-1
            for n = 1:4
                @inbounds U[i, j, n] = U[i, Ny+NG-1, n]
            end
        end
    end
    return
end

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

function time_step!(U, Time, dt, dξdx, dξdy, dηdx, dηdy, J, Nx, Ny, NG, debugger)
    Nx_tot = Nx+2*NG
    Ny_tot = Ny+2*NG
    Q = CuArray(zeros(Float64, Nx_tot, Ny_tot, 6))
    Fp_x = CuArray(zeros(Float64, Nx_tot, Ny_tot, 4))
    Fm_x = CuArray(zeros(Float64, Nx_tot, Ny_tot, 4))
    Fp_y = CuArray(zeros(Float64, Nx_tot, Ny_tot, 4))
    Fm_y = CuArray(zeros(Float64, Nx_tot, Ny_tot, 4))
    Fx = CuArray(zeros(Float64, Nx-1, Ny-2, 4))
    Fy = CuArray(zeros(Float64, Nx-2, Ny-1, 4))

    Fv_x = CuArray(zeros(Float64, Nx_tot-4, Ny_tot-4, 3))
    Fv_y = CuArray(zeros(Float64, Nx_tot-4, Ny_tot-4, 3))

    nthreads = (16, 16)
    nblock = (cld((Nx+2*NG), 16), 
              cld((Ny+2*NG), 16))

    # local constants
    gamma::Float64 = 1.4
    Ru::Float64 = 8.31446
    eos_m::Float64 = 28.97e-3
    Rg::Float64 = Ru/eos_m
    Pr::Float64 = 0.72
    Cv::Float64 = Rg/(gamma-1)
    Cp::Float64 = gamma * Cv
    C_s::Float64 = 1.458e-6
    T_s::Float64 = 110.4
    tmp0::Float64 = 1.0/(2*gamma)
    split_C1::Float64 = 2.0*(gamma-1)
    split_C3::Float64 = (3.0-gamma)/(2*(gamma-1))
    T_wall::Float64 = 300

    for tt = 1:cld(Time, dt)
        if tt % 10 == 0
            println("Step: $tt")
        end

        @cuda threads=nthreads blocks=nblock c2Prim!(U, Q, Nx, Ny, NG, gamma, Rg)

        @cuda threads=nthreads blocks=nblock fluxSplit!(Q, Fp_x, Fm_x, Nx, Ny, NG, dξdx, dξdy, gamma, tmp0, split_C1, split_C3)
        @cuda threads=nthreads blocks=nblock NND_x!(Fx, Fp_x, Fm_x, NG, Nx, Ny)

        @cuda threads=nthreads blocks=nblock fluxSplit!(Q, Fp_y, Fm_y, Nx, Ny, NG, dηdx, dηdy, gamma, tmp0, split_C1, split_C3)
        @cuda threads=nthreads blocks=nblock NND_y!(Fy, Fp_y, Fm_y, NG, Nx, Ny)

        @cuda threads=nthreads blocks=nblock viscousFlux!(Fv_x, Fv_y, Q, NG, Nx, Ny, Pr, Cp, C_s, T_s, dξdx, dξdy, dηdx, dηdy, J)

        @cuda threads=nthreads blocks=nblock div!(U, Fx, Fy, Fv_x, Fv_y, dt, NG, Nx, Ny, J)

        @cuda threads=nthreads blocks=nblock fillGhost!(U, NG, Nx, Ny, gamma, T_wall, Rg)

    end

    copyto!(debugger, Fv_y)
    return
end


@load "metrics.jld2" NG Nx Ny dξdx dξdy dηdx dηdy J x y

dt::Float64 = 1e-8
Time::Float64 = 2e-3

U = zeros(Float64, Nx+2*NG, Ny+2*NG, 4)
debugger = zeros(Float64, Nx+2*NG-4, Ny+2*NG-4, 3)

#initialization on CPU
# Mach 6 inlet
T = 79
P = 0.01 * 101325
c = sqrt(1.4 * 287 * T)
u = 6*c
ρ = P/(287 * T)

U[:, :, 1] .= ρ
U[:, :, 2] .= ρ * u
U[:, :, 3] .= 0.0
U[:, :, 4] .= P/(1.4-1) + 0.5*ρ*u^2


U_gpu = CuArray(U)
dξdx_d = CuArray(dξdx)
dξdy_d = CuArray(dξdy)
dηdx_d = CuArray(dηdx)
dηdy_d = CuArray(dηdy)
J_d = CuArray(J)

time_step!(U_gpu, Time, dt, dξdx_d, dξdy_d, dηdx_d, dηdy_d, J_d, Nx, Ny, NG, debugger)
copyto!(U, U_gpu)

# Debug
# x = x[3:Nx+2*NG-2, 3:Ny+2*NG-2]
# y = y[3:Nx+2*NG-2, 3:Ny+2*NG-2]
# Fv1 = debugger[:, :, 1]
# Fv2 = debugger[:, :, 2]
# Fv3 = debugger[:, :, 3]
# vtk_grid("result.vts", x, y) do vtk
#     vtk["Fv1"] = Fv1
#     vtk["Fv2"] = Fv2
#     vtk["Fv3"] = Fv3
# end

# remove ghost cells and write 
rho = U[NG+1:Nx+NG, NG+1:Ny+NG, 1]
u = U[NG+1:Nx+NG, NG+1:Ny+NG, 2]./rho
v = U[NG+1:Nx+NG, NG+1:Ny+NG, 3]./rho
p = (U[NG+1:Nx+NG, NG+1:Ny+NG, 4] - 0.5.*rho.*(u.^2+v.^2)) * 0.4
T = p./(287.0 .* rho)
x = x[NG+1:Nx+NG, NG+1:Ny+NG]
y = y[NG+1:Nx+NG, NG+1:Ny+NG]
vtk_grid("result.vts", x, y) do vtk
    vtk["rho"] = rho
    vtk["u"] = u
    vtk["v"] = v
    vtk["p"] = p
    vtk["T"] = T
end 
