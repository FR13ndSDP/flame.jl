using WriteVTK
using CUDA
using JLD2
CUDA.allowscalar(false)

@inline function minmod(a, b)
    if a*b > 0
        return (abs(a)>abs(b)) ? b : a
    end
    return 0
end

#Range: 1 -> N-1
function NND!(F, Fp, Fm, NG, Nx, Ny, dir)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y

    if dir == 'x'
        if i > Nx-1 || j > Ny-2
            return
        end
        for n = 1:4
            @inbounds fp = Fp[i+NG, j+1+NG, n] + 0.5*minmod(Fp[i+1+NG, j+1+NG, n]-Fp[i+NG, j+1+NG, n], Fp[i+NG, j+1+NG, n] - Fp[i-1+NG, j+1+NG, n])
            @inbounds fm = Fm[i+1+NG, j+1+NG, n] - 0.5*minmod(Fm[i+2+NG, j+1+NG, n]-Fm[i+1+NG, j+1+NG, n], Fm[i+1+NG, j+1+NG, n] - Fm[i+NG, j+1+NG, n])
            @inbounds F[i, j, n] = fp + fm
        end
    elseif dir == 'y'
        if i > Nx-2 || j > Ny-1
            return
        end
        for n = 1:4
            @inbounds fp = Fp[i+1+NG, j+NG, n] + 0.5*minmod(Fp[i+1+NG, j+1+NG, n]-Fp[i+1+NG, j+NG, n], Fp[i+1+NG, j+NG, n] - Fp[i+1+NG, j-1+NG, n])
            @inbounds fm = Fm[i+1+NG, j+1+NG, n] - 0.5*minmod(Fm[i+1+NG, j+2+NG, n]-Fm[i+1+NG, j+1+NG, n], Fm[i+1+NG, j+1+NG, n] - Fm[i+1+NG, j+NG, n])
            @inbounds F[i, j, n] = fp + fm
        end
    end
    return
end

# Range: 1 -> N+2*NG
function c2Prim!(U, Q, Nx, Ny, NG, gamma)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    if i > Nx+2*NG || j > Ny+2*NG
        return
    end

    @inbounds Q[i, j, 1] = U[i, j, 1] # ρ
    @inbounds Q[i, j, 2] = U[i, j, 2]/U[i, j, 1] # U
    @inbounds Q[i, j, 3] = U[i, j, 3]/U[i, j, 1] # V
    @inbounds Q[i, j, 4] = (gamma-1) * (U[i, j, 4] - 0.5*Q[i, j, 1]*(Q[i, j, 2]^2 + Q[i, j, 3]^2)) # P
    @inbounds Q[i, j, 5] = sqrt(gamma * Q[i, j, 4] / Q[i, j, 1]) # speed of sound
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
    @inbounds c = Q[i, j, 5]
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
    else
        return
    end
    return
end


function time_step!(U, T, dt, dξdx, dξdy, dηdx, dηdy, J, Nx, Ny, NG)
    Q = CuArray(zeros(Float64, Nx+2*NG, Ny+2*NG, 5))
    Fp_x = CuArray(zeros(Float64, Nx+2*NG, Ny+2*NG, 4))
    Fm_x = CuArray(zeros(Float64, Nx+2*NG, Ny+2*NG, 4))
    Fp_y = CuArray(zeros(Float64, Nx+2*NG, Ny+2*NG, 4))
    Fm_y = CuArray(zeros(Float64, Nx+2*NG, Ny+2*NG, 4))
    Fx = CuArray(zeros(Float64, Nx-1, Ny-2, 4))
    Fy = CuArray(zeros(Float64, Nx-2, Ny-1, 4))

    nthreads = (16, 16, 1)
    nblock = (cld((Nx+2*NG), 16), 
              cld((Ny+2*NG), 16), 1)

    gamma::Float64 = 1.4
    tmp0::Float64 = 1.0/(2*gamma)
    split_C1::Float64 = 2.0*(gamma-1)
    split_C3::Float64 = (3.0-gamma)/(2*(gamma-1))

    for tt = 1:(T/dt)
        @cuda threads=nthreads blocks=nblock c2Prim!(U, Q, Nx, Ny, NG, gamma)

        @cuda threads=nthreads blocks=nblock fluxSplit!(Q, Fp_x, Fm_x, Nx, Ny, NG, dξdx, dξdy, gamma, tmp0, split_C1, split_C3)
        @cuda threads=nthreads blocks=nblock NND!(Fx, Fp_x, Fm_x, NG, Nx, Ny, 'x')

        @cuda threads=nthreads blocks=nblock fluxSplit!(Q, Fp_y, Fm_y, Nx, Ny, NG, dηdx, dηdy, gamma, tmp0, split_C1, split_C3)
        @cuda threads=nthreads blocks=nblock NND!(Fy, Fp_y, Fm_y, NG, Nx, Ny, 'y')

        @cuda threads=nthreads blocks=nblock div!(U, Fx, Fy, dt, NG, Nx, Ny, J)

        @cuda threads=nthreads blocks=nblock fillGhost!(U, NG, Nx, Ny)

    end

    return
end


@load "metrics.jld2" NG Nx Ny dξdx dξdy dηdx dηdy J x y

dt::Float64 = 1f-4
T::Float64 = 0.4
gamma::Float64 = 1.4

U = zeros(Float64, Nx+2*NG, Ny+2*NG, 4)

#initialization on CPU
for j = 1:Ny+2*NG
    for i = 1:Nx+2*NG
        if i < (Nx+2*NG)/3
            U[i, j, 1] = 1.0
            U[i, j, 4] = 1.0/(gamma-1)
        else
            U[i, j, 1] = 0.125
            U[i, j, 4] = 0.1/(gamma-1)
        end
    end
end

U_gpu = CuArray(U)
dξdx_d = CuArray(dξdx)
dξdy_d = CuArray(dξdy)
dηdx_d = CuArray(dηdx)
dηdy_d = CuArray(dηdy)
J_d = CuArray(J)

time_step!(U_gpu, T, dt, dξdx_d, dξdy_d, dηdx_d, dηdy_d, J_d, Nx, Ny, NG)
copyto!(U, U_gpu)

# remove ghost cells and write 
rho = U[NG+1:NG+Nx, NG+1:NG+Ny, 1]
u = U[NG+1:NG+Nx, NG+1:NG+Ny, 2]./rho
v = U[NG+1:NG+Nx, NG+1:NG+Ny, 3]./rho
x = x[NG+1:NG+Nx, NG+1:NG+Ny]
y = y[NG+1:NG+Nx, NG+1:NG+Ny]
vtk_grid("result.vts", x, y) do vtk
    vtk["rho"] = rho
    vtk["u"] = u
    vtk["v"] = v
end 
