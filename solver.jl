using WriteVTK
using CUDA
using JLD2
CUDA.allowscalar(false)
include("schemes.jl")
include("viscous.jl")
include("boundary.jl")
include("utils.jl")


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

    ss = CUDA.sqrt(A1*A1 + A2*A2)
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

    vvc1 = (uc1*uc1 + vc1*vc1) * 0.50
    vvc2 = (uc2*uc2 + vc2*vc2) * 0.50
    vv = (gamma - 1.0) * (u*u + v*v)
    W2 = split_C3 * c * c

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

    c1::Float64 = -1/60
    c2::Float64 = 3/20
    c3::Float64 = -3/4

    @inbounds Jac = J[i, j]
    @inbounds dV11dξ = c1*(Fv_x[i-1-NG,j+2-NG,1] - Fv_x[i+5-NG, j+2-NG, 1]) + c2*(Fv_x[i-NG,j+2-NG, 1] - Fv_x[i+4-NG, j+2-NG, 1]) + c3*(Fv_x[i+1-NG,j+2-NG, 1] - Fv_x[i+3-NG, j+2-NG, 1])
    @inbounds dV12dξ = c1*(Fv_x[i-1-NG,j+2-NG,2] - Fv_x[i+5-NG, j+2-NG, 2]) + c2*(Fv_x[i-NG,j+2-NG, 2] - Fv_x[i+4-NG, j+2-NG, 2]) + c3*(Fv_x[i+1-NG,j+2-NG, 2] - Fv_x[i+3-NG, j+2-NG, 2])
    @inbounds dV13dξ = c1*(Fv_x[i-1-NG,j+2-NG,3] - Fv_x[i+5-NG, j+2-NG, 3]) + c2*(Fv_x[i-NG,j+2-NG, 3] - Fv_x[i+4-NG, j+2-NG, 3]) + c3*(Fv_x[i+1-NG,j+2-NG, 3] - Fv_x[i+3-NG, j+2-NG, 3])

    @inbounds dV21dη = c1*(Fv_y[i+2-NG,j-1-NG,1] - Fv_y[i+2-NG, j+5-NG, 1]) + c2*(Fv_y[i+2-NG,j-NG, 1] - Fv_y[i+2-NG, j+4-NG, 1]) + c3*(Fv_y[i+2-NG,j+1-NG, 1] - Fv_y[i+2-NG, j+3-NG, 1])
    @inbounds dV22dη = c1*(Fv_y[i+2-NG,j-1-NG,2] - Fv_y[i+2-NG, j+5-NG, 2]) + c2*(Fv_y[i+2-NG,j-NG, 2] - Fv_y[i+2-NG, j+4-NG, 2]) + c3*(Fv_y[i+2-NG,j+1-NG, 2] - Fv_y[i+2-NG, j+3-NG, 2])
    @inbounds dV23dη = c1*(Fv_y[i+2-NG,j-1-NG,3] - Fv_y[i+2-NG, j+5-NG, 3]) + c2*(Fv_y[i+2-NG,j-NG, 3] - Fv_y[i+2-NG, j+4-NG, 3]) + c3*(Fv_y[i+2-NG,j+1-NG, 3] - Fv_y[i+2-NG, j+3-NG, 3])

    for n = 1:4
        @inbounds U[i, j, n] +=  (Fx[i-1-NG, j-1-NG, n] - Fx[i-NG, j-1-NG, n] + Fy[i-1-NG, j-1-NG, n] - Fy[i-1-NG, j-NG, n]) * dt * Jac
    end
    @inbounds U[i, j, 2] += (dV11dξ + dV21dη) * dt * Jac
    @inbounds U[i, j, 3] += (dV12dξ + dV22dη) * dt * Jac
    @inbounds U[i, j, 4] += (dV13dξ + dV23dη) * dt * Jac
    return
end


function RHS(U, Q, Fp_x, Fm_x, Fp_y, Fm_y, Fx, Fy, Fv_x, Fv_y, Nx, Ny, NG, dξdx, dξdy, dηdx, dηdy, J)
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

    @cuda fastmath=true threads=nthreads blocks=nblock c2Prim!(U, Q, Nx, Ny, NG, gamma, Rg)

    @cuda fastmath=true threads=nthreads blocks=nblock fluxSplit!(Q, Fp_x, Fm_x, Nx, Ny, NG, dξdx, dξdy, gamma, tmp0, split_C1, split_C3)
    @cuda fastmath=true threads=nthreads blocks=nblock WENO_x!(Fx, Fp_x, Fm_x, NG, Nx, Ny)

    @cuda fastmath=true threads=nthreads blocks=nblock fluxSplit!(Q, Fp_y, Fm_y, Nx, Ny, NG, dηdx, dηdy, gamma, tmp0, split_C1, split_C3)
    @cuda fastmath=true threads=nthreads blocks=nblock WENO_y!(Fy, Fp_y, Fm_y, NG, Nx, Ny)

    @cuda fastmath=true threads=nthreads blocks=nblock viscousFlux!(Fv_x, Fv_y, Q, NG, Nx, Ny, Pr, Cp, C_s, T_s, dξdx, dξdy, dηdx, dηdy, J)
end

function time_step!(U, Time, dt, dξdx, dξdy, dηdx, dηdy, J, Nx, Ny, NG)
    Nx_tot = Nx+2*NG
    Ny_tot = Ny+2*NG
    Un = CuArray(zeros(Float64, Nx_tot, Ny_tot, 4))
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

    for tt = 1:cld(Time, dt)
        if tt % 100 == 0
            printstyled("Step: ", color=:cyan)
            print("$tt")
            printstyled("\tTime: ", color=:blue)
            println("$(tt*dt)")
            if any(isnan, U_gpu)
                printstyled("Oops, NaN detected\n", color=:red)
                return
            end
        end

        # RK3-1
        @cuda threads=nthreads blocks=nblock copyOld!(Un, U, Nx, Ny, NG)

        RHS(U, Q, Fp_x, Fm_x, Fp_y, Fm_y, Fx, Fy, Fv_x, Fv_y, Nx, Ny, NG, dξdx, dξdy, dηdx, dηdy, J)

        @cuda threads=nthreads blocks=nblock div!(U, Fx, Fy, Fv_x, Fv_y, dt, NG, Nx, Ny, J)

        @cuda threads=nthreads blocks=nblock fillGhost!(U, NG, Nx, Ny)

        # RK3-2
        RHS(U, Q, Fp_x, Fm_x, Fp_y, Fm_y, Fx, Fy, Fv_x, Fv_y, Nx, Ny, NG, dξdx, dξdy, dηdx, dηdy, J)

        @cuda threads=nthreads blocks=nblock div!(U, Fx, Fy, Fv_x, Fv_y, dt, NG, Nx, Ny, J)

        @cuda threads=nthreads blocks=nblock linComb!(U, Un, Nx, Ny, NG, 0.25, 0.75)

        @cuda threads=nthreads blocks=nblock fillGhost!(U, NG, Nx, Ny)

        # RK3-3
        RHS(U, Q, Fp_x, Fm_x, Fp_y, Fm_y, Fx, Fy, Fv_x, Fv_y, Nx, Ny, NG, dξdx, dξdy, dηdx, dηdy, J)

        @cuda threads=nthreads blocks=nblock div!(U, Fx, Fy, Fv_x, Fv_y, dt, NG, Nx, Ny, J)

        @cuda threads=nthreads blocks=nblock linComb!(U, Un, Nx, Ny, NG, 2/3, 1/3)

        @cuda threads=nthreads blocks=nblock fillGhost!(U, NG, Nx, Ny)
    end
    return
end
