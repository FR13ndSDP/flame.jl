using WriteVTK
using CUDA
using JLD2
CUDA.allowscalar(false)

@inline function minmod(a, b)
    ifelse(a*b > 0, (abs(a)>abs(b)) ? b : a, 0)
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
    @inbounds Q[i, j, 6] = sqrt(gamma * Q[i, j, 4] / Q[i, j, 1]) # speed of sound
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

    ss = sqrt(A1^2 + A2^2)
    E1 = A1*u + A2*v
    E2 = E1 - c*ss
    E3 = E1 + c*ss

    ss = 1/ss

    A1 *= ss
    A2 *= ss

    E1P = (E1 + sqrt(E1 * E1)) * 0.5
    E2P = (E2 + sqrt(E2 * E2)) * 0.5
    E3P = (E3 + sqrt(E3 * E3)) * 0.5

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
function div!(U, Fx, Fy, dt, NG, Nx, Ny, J)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    if i > Nx+NG-1 || i < 2+NG || j > Ny+NG-1 || j < 2+NG
        return
    end

    @inbounds Jac = J[i, j]
    for n = 1:4
        @inbounds U[i, j, n] +=  (Fx[i-1-NG, j-1-NG, n] - Fx[i-NG, j-1-NG, n] + Fy[i-1-NG, j-1-NG, n] - Fy[i-1-NG, j-NG, n]) * dt * Jac
    end
    return
end

function fillGhost!(U, NG, Nx, Ny)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    
    if i > Nx+2*NG || j > Ny+2*NG
        return
    end

    if j <= NG+1 
        @inbounds U[i, j, 1] = U[i, 7-j, 1]
        @inbounds U[i, j, 2] = 0
        @inbounds U[i, j, 3] = 0
        @inbounds U[i, j, 4] = U[i, 7-j, 4] - 0.5/U[i, 7-j, 1]*(U[i, 7-j, 2]^2 + U[i, 7-j, 3]^2)
    elseif j > Ny+NG-1
        for n = 1:4
            @inbounds U[i, j, n] = U[i, Ny+NG-1, n]
        end
    end

    if i <= 1+NG
        @inbounds U[i, j, 1] = 0.4468971904908923
        @inbounds U[i, j, 2] = 0.4468971904908923 * 534.4902244194930
        @inbounds U[i, j, 3] = 0.0
        @inbounds U[i, j, 4] = 0.1*101325/0.4 + 0.5*0.4468971904908923*534.4902244194930^2
    elseif i > Nx + NG -1
        for n = 1:4
            @inbounds U[i, j, n] = U[Nx+NG-1, j, n]
        end
    end
    return
end

# #Range: 1 -> N-1
# function viscousFlux_x!(Fx, Fy, Q, NG, Nx, Ny, Pr, Cp, C_s, T_s, dξdx, dξdy, dηdx, dηdy, J)
#     i = (blockIdx().x-1)* blockDim().x + threadIdx().x
#     j = (blockIdx().y-1)* blockDim().y + threadIdx().y

#     if i > Nx-1 || j > Ny-2
#         return
#     end

#     @inbounds Tl = Q[i+NG, j+1+NG, 5]
#     @inbounds Tr = Q[i+1+NG, j+1+NG, 5]
#     T = 0.5*(Tl + Tr)
#     μ = C_s * T * sqrt(T)/(T + T_s)
#     k = Cp*μ/Pr
#     dTdx = Tr - Tl
#     @inbounds dudx = Q[i+1+NG, j+1+NG, 2]-Q[i+NG, j+1+NG, 2]
#     @inbounds dudy = 0.5 * (Q[i+1+NG, j+2+NG, 2]+Q[i+NG, j+2+NG, 2]-Q[i+1+NG, j+NG, 2]-Q[i+NG, j+NG, 2])
#     @inbounds dvdx = Q[i+1+NG, j+1+NG, 3]-Q[i+NG, j+1+NG, 3]
#     @inbounds dvdy = 0.5 * (Q[i+1+NG, j+2+NG, 3]+Q[i+NG, j+2+NG, 3]-Q[i+1+NG, j+NG, 3]-Q[i+NG, j+NG, 3])
#     τ11 = μ*(4/3*dudx - 2/3*dvdy)
#     τ12 = μ*(dudy + dvdx)
#     @inbounds Fx[i, j, 2] += τ11
#     @inbounds Fx[i, j, 3] += τ12
#     @inbounds Fx[i, j, 4] += (0.5 * (Q[i+NG, j+1+NG, 2] + Q[i+1+NG, j+1+NG, 2]) * τ11
#                   + 0.5 * (Q[i+NG, j+1+NG, 3] + Q[i+1+NG, j+1+NG, 3]) * τ12
#                   + k * dTdx)
#     return
# end

# function viscousFlux_y!(Fy, Q, NG, Nx, Ny, Pr, Cp, C_s, T_s)
#     i = (blockIdx().x-1)* blockDim().x + threadIdx().x
#     j = (blockIdx().y-1)* blockDim().y + threadIdx().y

#     if j > Ny-1 || i > Nx-2
#         return
#     end

#     @inbounds Tl = Q[i+1+NG, j+NG, 5]
#     @inbounds Tr = Q[i+1+NG, j+1+NG, 5]
#     T = 0.5*(Tl + Tr)
#     μ = C_s * T * sqrt(T)/(T + T_s)
#     k = Cp*μ/Pr
#     dTdy = Tr - Tl
#     @inbounds dudx = 0.5 * (Q[i+2+NG, j+NG, 2]+Q[i+2+NG, j+1+NG, 2]-Q[i+NG, j+NG, 2]-Q[i+NG, j+1+NG, 2])
#     @inbounds dudy = Q[i+1+NG, j+1+NG, 2]-Q[i+1+NG, j+NG, 2]
#     @inbounds dvdx = 0.5 * (Q[i+2+NG, j+NG, 3]+Q[i+2+NG, j+1+NG, 3]-Q[i+NG, j+NG, 3]-Q[i+NG, j+1+NG, 3])
#     @inbounds dvdy = Q[i+1+NG, j+1+NG, 3]-Q[i+1+NG, j+NG, 3]
#     τ21 = μ*(dudy + dvdx)
#     τ22 = μ*(4/3*dvdy - 2/3*dudx)
#     @inbounds Fy[i, j, 2] += τ21
#     @inbounds Fy[i, j, 3] += τ22
#     @inbounds Fy[i, j, 4] += (0.5 * (Q[i+1+NG, j+NG, 2] + Q[i+1+NG, j+1+NG, 2]) * τ21
#                   + 0.5 * (Q[i+1+NG, j+NG, 3] + Q[i+1+NG, j+1+NG, 3]) * τ22
#                   + k * dTdy)
#     return
# end

function time_step!(U, Time, dt, dξdx, dξdy, dηdx, dηdy, J, Nx, Ny, NG)
    Q = CuArray(zeros(Float64, Nx+2*NG, Ny+2*NG, 6))
    Fp_x = CuArray(zeros(Float64, Nx+2*NG, Ny+2*NG, 4))
    Fm_x = CuArray(zeros(Float64, Nx+2*NG, Ny+2*NG, 4))
    Fp_y = CuArray(zeros(Float64, Nx+2*NG, Ny+2*NG, 4))
    Fm_y = CuArray(zeros(Float64, Nx+2*NG, Ny+2*NG, 4))
    Fx = CuArray(zeros(Float64, Nx-1, Ny-2, 4))
    Fy = CuArray(zeros(Float64, Nx-2, Ny-1, 4))

    Fv_x = CuArray(zeros(Float64, Nx-1, Ny-2, 4))
    Fv_y = CuArray(zeros(Float64, Nx-2, Ny-1, 4))

    nthreads = (16, 16, 1)
    nblock = (cld((Nx+2*NG), 16), 
              cld((Ny+2*NG), 16), 1)

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

    for tt = 1:cld(Time, dt)
        if tt % 10 == 0
            println("Step: $tt")
        end

        @cuda threads=nthreads blocks=nblock c2Prim!(U, Q, Nx, Ny, NG, gamma, Rg)

        @cuda threads=nthreads blocks=nblock fluxSplit!(Q, Fp_x, Fm_x, Nx, Ny, NG, dξdx, dξdy, gamma, tmp0, split_C1, split_C3)
        @cuda threads=nthreads blocks=nblock NND_x!(Fx, Fp_x, Fm_x, NG, Nx, Ny)

        @cuda threads=nthreads blocks=nblock fluxSplit!(Q, Fp_y, Fm_y, Nx, Ny, NG, dηdx, dηdy, gamma, tmp0, split_C1, split_C3)
        @cuda threads=nthreads blocks=nblock NND_y!(Fy, Fp_y, Fm_y, NG, Nx, Ny)

        # @cuda threads=nthreads blocks=nblock viscousFlux!(Fx, Fy, Q, NG, Nx, Ny, Pr, Cp, C_s, T_s, dξdx, dξdy, dηdx, dηdy, J)

        @cuda threads=nthreads blocks=nblock div!(U, Fx, Fy, dt, NG, Nx, Ny, J)

        @cuda threads=nthreads blocks=nblock fillGhost!(U, NG, Nx, Ny)

    end

    return
end


@load "metrics.jld2" NG Nx Ny dξdx dξdy dηdx dηdy J x y

dt::Float64 = 1e-7
Time::Float64 = 1e-2
gamma::Float64 = 1.4

U = zeros(Float64, Nx+2*NG, Ny+2*NG, 4)

#initialization on CPU
T = 79.0
P = 0.1 * 101325
c = sqrt(gamma * 287 * T)
u = 3*c
ρ = P/(287 * T)
T_wall = 300.0

U[:, :, 1] .= ρ
U[:, :, 2] .= ρ * u
U[:, :, 3] .= 0.0
U[:, :, 4] .= P/(gamma-1) + 0.5*ρ*u^2


U_gpu = CuArray(U)
dξdx_d = CuArray(dξdx)
dξdy_d = CuArray(dξdy)
dηdx_d = CuArray(dηdx)
dηdy_d = CuArray(dηdy)
J_d = CuArray(J)

time_step!(U_gpu, Time, dt, dξdx_d, dξdy_d, dηdx_d, dηdy_d, J_d, Nx, Ny, NG)
copyto!(U, U_gpu)

# remove ghost cells and write 
rho = U[:, :, 1]
u = U[:, :, 2]./rho
v = U[:, :, 3]./rho
vtk_grid("result.vts", x, y) do vtk
    vtk["rho"] = rho
    vtk["u"] = u
    vtk["v"] = v
end 
