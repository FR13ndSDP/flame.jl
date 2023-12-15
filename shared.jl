using CUDA
CUDA.allowscalar(false)

# struct Params
#     Ru::Float64
#     eos_m::Float64
#     Rg::Float64
#     Pr::Float64
#     C_s::Float64
#     T_s::Float64
#     gamma::Float64
#     tmp0::Float64
#     split_C1::Float64
#     split_C3::Float64
# end

# const global param = Params(8.31446, 
#                             28.97e-3, 
#                             8.31446/28.97e-3, 
#                             0.72, 
#                             1.458e-6, 
#                             110.4, 
#                             1.4, 
#                             1/(2*1.4), 
#                             0.8, 
#                             2.0)
# param = Mem.pin(param)


#Range: 1->N+1
function vanLeer!(F, QL, QR, Nx, Ny, dir)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    if i > Nx+1 || j > Ny+1
        return
    end

    gamma::Float64 = 1.4

    @inbounds ρL = QL[i,j,1]
    @inbounds uL = QL[i,j,2]
    @inbounds vL = QL[i,j,3]
    @inbounds PL = QL[i,j,4]
    @inbounds cL = QL[i,j,5]
    @inbounds ρR = QR[i,j,1]
    @inbounds uR = QR[i,j,2]
    @inbounds vR = QR[i,j,3]
    @inbounds PR = QR[i,j,4]
    @inbounds cR = QR[i,j,5]

    if dir == 'x'
        ML = uL/cL
        MR = uR/cR
    else
        ML = vL/cL
        MR = vR/cR
    end

    if ML >= 1
        ML⁺ = ML
    elseif abs(ML) < 1
        ML⁺ = 0.25*(ML+1)^2
    else
        ML⁺ = 0
    end

    if MR >= 1
        MR⁻ = 0
    elseif abs(MR) < 1
        MR⁻ = -0.25*(MR-1)^2
    else
        MR⁻ = MR
    end

    Mn = ML⁺ + MR⁻
    if dir == 'x'
        if Mn >= 1
            @inbounds F[i, j, 1] = ρL * uL
            @inbounds F[i, j, 2] = ρL * uL * uL + PL
            @inbounds F[i, j, 3] = ρL * vL * uL
            @inbounds F[i, j, 4] = uL*(gamma*PL/(gamma-1) + 0.5*ρL*(uL^2 + vL^2))
        elseif Mn <= -1
            @inbounds F[i, j, 1] = ρR * uR
            @inbounds F[i, j, 2] = ρR * uR * uR + PR
            @inbounds F[i, j, 3] = ρR * vR * uR
            @inbounds F[i, j, 4] = uR*(gamma*PR/(gamma-1) + 0.5*ρR*(uR^2 + vR^2))
        else
            fm⁺ = ρL * cL * (ML+1)^2/4
            fm⁻ = -ρR * cR * (MR-1)^2/4
            fe⁺ = fm⁺*(((gamma-1)*uL+2*cL)^2/(2*(gamma^2-1)) + vL^2/2)
            fe⁻ = fm⁻*(((gamma-1)*uR-2*cR)^2/(2*(gamma^2-1)) + vR^2/2)
            fu⁺ = fm⁺*((-uL+2*cL)/gamma + uL)
            fu⁻ = fm⁻*((-uR-2*cR)/gamma + uR)
            fv⁺ = fm⁺*vL
            fv⁻ = fm⁻*vR
            @inbounds F[i, j, 1] = fm⁺ + fm⁻
            @inbounds F[i, j, 2] = fu⁺ + fu⁻
            @inbounds F[i, j, 3] = fv⁺ + fv⁻
            @inbounds F[i, j, 4] = fe⁺ + fe⁻
        end
    else
        if Mn >= 1
            @inbounds F[i, j, 1] = ρL * vL
            @inbounds F[i, j, 2] = ρL * uL * vL
            @inbounds F[i, j, 3] = ρL * vL * vL + PL
            @inbounds F[i, j, 4] = vL*(gamma*PL/(gamma-1) + 0.5*ρL*(uL^2 + vL^2))
        elseif Mn <= -1
            @inbounds F[i, j, 1] = ρR * vR
            @inbounds F[i, j, 2] = ρR * uR * vR
            @inbounds F[i, j, 3] = ρR * vR * vR + PR
            @inbounds F[i, j, 4] = vR*(gamma*PR/(gamma-1) + 0.5*ρR*(uR^2 + vR^2))
        else
            fm⁺ = ρL * cL * (ML+1)^2/4
            fm⁻ = -ρR * cR * (MR-1)^2/4
            fe⁺ = fm⁺*(((gamma-1)*vL+2*cL)^2/(2*(gamma^2-1)) + uL^2/2)
            fe⁻ = fm⁻*(((gamma-1)*vR-2*cR)^2/(2*(gamma^2-1)) + uR^2/2)
            fv⁺ = fm⁺*((-vL+2*cL)/gamma + vL)
            fv⁻ = fm⁻*((-vR-2*cR)/gamma + vR)
            fu⁺ = fm⁺*uL
            fu⁻ = fm⁻*uR
            @inbounds F[i, j, 1] = fm⁺ + fm⁻
            @inbounds F[i, j, 2] = fu⁺ + fu⁻
            @inbounds F[i, j, 3] = fv⁺ + fv⁻
            @inbounds F[i, j, 4] = fe⁺ + fe⁻
        end
    end
    return
end

# RK3-1
@cuda threads=nthreads blocks=nblock copyOld!(Un, U, Nx, Ny, NG)

specAdvance(ρi, Fp_i, Fm_i, Fx_i, Fy_i, Nx, Ny, NG, dξdx, dξdy, dηdx, dηdy, J, thermo)

RHS(U, ρi, Q, Fp, Fm, Fx, Fy, Fv_x, Fv_y, Nx, Ny, NG, dξdx, dξdy, dηdx, dηdy, J, thermo)

@cuda threads=nthreads blocks=nblock div(U, Fx, Fy, Fv_x, Fv_y, dt, NG, Nx, Ny, J)

@cuda threads=nthreads blocks=nblock fillGhost(U, NG, Nx, Ny)

# RK3-2
RHS(U, ρi, Q, Fp_x, Fm_x, Fp_y, Fm_y, Fx, Fy, Fv_x, Fv_y, Nx, Ny, NG, dξdx, dξdy, dηdx, dηdy, J)

@cuda threads=nthreads blocks=nblock div!(U, Fx, Fy, Fv_x, Fv_y, dt, NG, Nx, Ny, J)

@cuda threads=nthreads blocks=nblock linComb!(U, Un, Nx, Ny, NG, 0.25, 0.75)

@cuda threads=nthreads blocks=nblock fillGhost!(U, NG, Nx, Ny)

# RK3-3
RHS(U, ρi, Q, Fp_x, Fm_x, Fp_y, Fm_y, Fx, Fy, Fv_x, Fv_y, Nx, Ny, NG, dξdx, dξdy, dηdx, dηdy, J)

@cuda threads=nthreads blocks=nblock div!(U, Fx, Fy, Fv_x, Fv_y, dt, NG, Nx, Ny, J)

@cuda threads=nthreads blocks=nblock linComb!(U, Un, Nx, Ny, NG, 2/3, 1/3)

@cuda threads=nthreads blocks=nblock fillGhost!(U, NG, Nx, Ny)


# Range: 1 -> N+2*NG
function c2Prim(U, ρi, Q, Nx, Ny, NG, thermo)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    if i > Nx+2*NG || j > Ny+2*NG
        return
    end

    rhoi = CuStaticSharedArray(Float64, Nspecs)

    for n = 1:Nspecs
        @inbounds rhoi[n] = ρi[i, j, n]
    end
    @inbounds ρ = U[i, j, 1] # ρ
    @inbounds u = U[i, j, 2]/ρ # U
    @inbounds v = U[i, j, 3]/ρ # V
    @inbounds ei = U[i, j, 4] - 0.5*ρ*(u^2 + v^2)

    T = GetT(ei, rhoi, thermo)
    p = Pmixture(T, rhoi, thermo)
    γ = p/ei + 1

    @inbounds Q[i, j, 1] = ρ
    @inbounds Q[i, j, 2] = u
    @inbounds Q[i, j, 3] = v
    @inbounds Q[i, j, 4] = p
    @inbounds Q[i, j, 5] = T
    @inbounds Q[i, j, 6] = CUDA.sqrt(γ * p / ρ)
    return
end

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

# compute jacobian
function CD6(f)
    fₓ = 1/60*(f[7]-f[1]) - 3/20*(f[6]-f[2]) + 3/4*(f[5]-f[3])
    return fₓ
end

function CD3_L(f)
    fₓ = -11/6*f[1] + 3*f[2] - 1.5*f[3] + 1/3*f[4]
    return fₓ
end

function CD3_R(f)
    fₓ = -1/3*f[1] + 1.5*f[2] - 3*f[3] + 11/6*f[4]
    return fₓ
end