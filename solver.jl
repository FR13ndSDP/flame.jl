using WriteVTK
using CUDA
using JLD2, JSON
using Lux, LuxCUDA

CUDA.allowscalar(false)
include("split.jl")
include("schemes.jl")
include("viscous.jl")
include("boundary.jl")
include("utils.jl")
include("div.jl")

function correction(U, ρi, NG, Nx, Ny)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    if i > Nx+2*NG || j > Ny+2*NG
        return
    end

    @inbounds ρ = U[i, j, 1]
    ∑ρ = 0
    for n = 1:Nspecs
        @inbounds ∑ρ += ρi[i, j, n]
    end
    for n = 1:Nspecs
        @inbounds ρi[i, j, n] *= ρ/∑ρ
    end
end

function flowAdvance(U, ρi, Q, Fp, Fm, Fx, Fy, Fv_x, Fv_y, Nx, Ny, NG, dξdx, dξdy, dηdx, dηdy, J, dt)
    nthreads = (16, 16, 1)
    nblock = (cld((Nx+2*NG), 16), 
              cld((Ny+2*NG), 16),
              1)

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

    @cuda threads=nthreads blocks=nblock fluxSplit(Q, U, Fp, Fm, Nx, Ny, NG, dξdx, dξdy)
    @cuda maxregs=32 threads=nthreads blocks=nblock NND_x(Fx, Fp, Fm, NG, Nx, Ny, Ncons)

    @cuda threads=nthreads blocks=nblock fluxSplit(Q, U, Fp, Fm, Nx, Ny, NG, dηdx, dηdy)
    @cuda maxregs=32 threads=nthreads blocks=nblock NND_y(Fy, Fp, Fm, NG, Nx, Ny, Ncons)

    @cuda threads=nthreads blocks=nblock viscousFlux(Fv_x, Fv_y, Q, NG, Nx, Ny, Pr, Cp, C_s, T_s, dξdx, dξdy, dηdx, dηdy, J)

    @cuda threads=nthreads blocks=nblock div(U, Fx, Fy, Fv_x, Fv_y, dt, NG, Nx, Ny, J)

end


function specAdvance(U, ρi, Q, Fp_i, Fm_i, Fx_i, Fy_i, Nx, Ny, NG, dξdx, dξdy, dηdx, dηdy, J, dt)
    nthreads = (16, 16, 1)
    nblock = (cld((Nx+2*NG), 16), 
              cld((Ny+2*NG), 16),
              1)

    @cuda threads=nthreads blocks=nblock c2Prim(U, Q, Nx, Ny, NG, 1.4, 287)

    @cuda threads=nthreads blocks=nblock split(ρi, Q, U, Fp_i, Fm_i, dξdx, dξdy, Nx, Ny, NG)
    @cuda maxregs=32 threads=nthreads blocks=nblock NND_x(Fx_i, Fp_i, Fm_i, NG, Nx, Ny, Nspecs)

    @cuda threads=nthreads blocks=nblock split(ρi, Q, U, Fp_i, Fm_i, dηdx, dηdy, Nx, Ny, NG)
    @cuda maxregs=32 threads=nthreads blocks=nblock NND_y(Fy_i, Fp_i, Fm_i, NG, Nx, Ny, Nspecs)

    @cuda threads=nthreads blocks=nblock divSpecs(ρi, Fx_i, Fy_i, dt, NG, Nx, Ny, J)
end

# Collect input
function pre_input(inputs, inputs_norm, Q, ρi, lambda, inputs_mean, inputs_std, Nx, Ny, NG)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y

    if i > Nx+2*NG || j > Ny+2*NG
        return
    end

    @inbounds inputs[1, (i-1)*(Ny+2*NG) + j] = Q[i, j, 5]
    @inbounds inputs[2, (i-1)*(Ny+2*NG) + j] = Q[i, j, 4]

    for n = 3:9
        @inbounds Yi = max(ρi[i, j, n-2], 0)/Q[i, j, 1]
        @inbounds inputs[n, (i-1)*(Ny+2*NG) + j] = (Yi^lambda - 1) / lambda
    end

    for n = 1:9
        @inbounds inputs_norm[n, (i-1)*(Ny+2*NG)+j] = (inputs[n, (i-1)*(Ny+2*NG)+j] - inputs_mean[n]) / inputs_std[n]
    end
    return
end

# Parse prediction
function post_predict(yt_pred, inputs, U, Q, ρi, dt, lambda, Nx, Ny, NG)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y

    if i > Nx+2*NG || j > Ny+2*NG
        return
    end

    @inbounds T = yt_pred[8, (i-1)*(Ny+2*NG)+j] * dt * Q[i, j, 5] + Q[i, j, 5]
    @inbounds U[i, j, 4] = Q[i, j, 1] * 287 * T /0.4 + 0.5*Q[i, j, 1] * (Q[i, j, 2]^2 + Q[i, j, 3]^2)
    for n = 1:Nspecs-1
        @inbounds Yi = (lambda * (yt_pred[n, (i-1)*(Ny+2*NG)+j] * dt + inputs[n+2, (i-1)*(Ny+2*NG)+j]) + 1) ^ (1/lambda)
        @inbounds ρi[i, j, n] = Yi * Q[i, j, 1]
    end
    return
end

function time_step(U, ρi, dξdx, dξdy, dηdx, dηdy, J, Nx, Ny, NG, dt)
    Nx_tot = Nx+2*NG
    Ny_tot = Ny+2*NG

    Un = copy(U)
    ρn = copy(ρi)
    Q =    CUDA.zeros(Float64, Nx_tot, Ny_tot, Nprim)
    Fp =   CUDA.zeros(Float64, Nx_tot, Ny_tot, Ncons)
    Fm =   CUDA.zeros(Float64, Nx_tot, Ny_tot, Ncons)
    Fx =   CUDA.zeros(Float64, Nx-1, Ny-2, Ncons)
    Fy =   CUDA.zeros(Float64, Nx-2, Ny-1, Ncons)
    Fv_x = CUDA.zeros(Float64, Nx_tot-4, Ny_tot-4, 3)
    Fv_y = CUDA.zeros(Float64, Nx_tot-4, Ny_tot-4, 3)
    Fp_i = CUDA.zeros(Float64, Nx_tot, Ny_tot, Nspecs)
    Fm_i = CUDA.zeros(Float64, Nx_tot, Ny_tot, Nspecs)
    Fx_i = CUDA.zeros(Float64, Nx-1, Ny-2, Nspecs)
    Fy_i = CUDA.zeros(Float64, Nx-2, Ny-1, Nspecs)

    nthreads = (16, 16, 1)
    nblock = (cld((Nx+2*NG), 16), 
              cld((Ny+2*NG), 16),
              1)


    @load "./NN/luxmodel.jld2" model ps st

    ps = ps |> gpu_device()

    j = JSON.parsefile("./NN/norm.json")
    lambda = j["lambda"]
    inputs_mean = CuArray(convert(Vector{Float64}, j["inputs_mean"]))
    inputs_std =  CuArray(convert(Vector{Float64}, j["inputs_std"]))
    labels_mean = CuArray(convert(Vector{Float64}, j["labels_mean"]))
    labels_std =  CuArray(convert(Vector{Float64}, j["labels_std"]))

    inputs = CUDA.zeros(Float64, 9, Nx_tot*Ny_tot)
    inputs_norm = CUDA.zeros(Float64, 9, Nx_tot*Ny_tot)

    dt1 = dt
    dt2 = dt/2

    for tt ∈ 1:ceil(Int, Time/dt)
        if tt % 100 == 0
            printstyled("Step: ", color=:cyan)
            print("$tt")
            printstyled("\tTime: ", color=:blue)
            println("$(tt*dt)")
            if any(isnan, U)
                printstyled("Oops, NaN detected\n", color=:red)
                return
            end
        end
        
        # Reaction Step
        @cuda threads=nthreads blocks=nblock c2Prim(U, Q, Nx, Ny, NG, 1.4, 287)
        @cuda threads=nthreads blocks=nblock pre_input(inputs, inputs_norm, Q, ρi, lambda, inputs_mean, inputs_std, Nx, Ny, NG)
        yt_pred = model(cu(inputs_norm), ps, st)[1]
        @. yt_pred = yt_pred * labels_std + labels_mean
        @cuda threads=nthreads blocks=nblock post_predict(yt_pred, inputs, U, Q, ρi, dt2, lambda, Nx, Ny, NG)
        @cuda threads=nthreads blocks=nblock fillSpec(ρi, U, NG, Nx, Ny)
        @cuda threads=nthreads blocks=nblock fillGhost(U, NG, Nx, Ny)
        @cuda threads=nthreads blocks=nblock correction(U, ρi, NG, Nx, Ny)

        # RK3-1
        @cuda threads=nthreads blocks=nblock copyOld(Un, U, Nx, Ny, NG, Ncons)
        @cuda threads=nthreads blocks=nblock copyOld(ρn, ρi, Nx, Ny, NG, Nspecs)

        specAdvance(U, ρi, Q, Fp_i, Fm_i, Fx_i, Fy_i, Nx, Ny, NG, dξdx, dξdy, dηdx, dηdy, J, dt1)

        flowAdvance(U, ρi, Q, Fp, Fm, Fx, Fy, Fv_x, Fv_y, Nx, Ny, NG, dξdx, dξdy, dηdx, dηdy, J, dt1)

        @cuda threads=nthreads blocks=nblock fillSpec(ρi, U, NG, Nx, Ny)
        @cuda threads=nthreads blocks=nblock fillGhost(U, NG, Nx, Ny)

        # RK3-2
        specAdvance(U, ρi, Q, Fp_i, Fm_i, Fx_i, Fy_i, Nx, Ny, NG, dξdx, dξdy, dηdx, dηdy, J, dt1)

        flowAdvance(U, ρi, Q, Fp, Fm, Fx, Fy, Fv_x, Fv_y, Nx, Ny, NG, dξdx, dξdy, dηdx, dηdy, J, dt1)

        @cuda threads=nthreads blocks=nblock linComb(U, Un, Nx, Ny, NG, Ncons, 0.25, 0.75)
        @cuda threads=nthreads blocks=nblock linComb(ρi, ρn, Nx, Ny, NG, Nspecs, 0.25, 0.75)
        @cuda threads=nthreads blocks=nblock fillSpec(ρi, U, NG, Nx, Ny)
        @cuda threads=nthreads blocks=nblock fillGhost(U, NG, Nx, Ny)

        # RK3-3
        specAdvance(U, ρi, Q, Fp_i, Fm_i, Fx_i, Fy_i, Nx, Ny, NG, dξdx, dξdy, dηdx, dηdy, J, dt1)

        flowAdvance(U, ρi, Q, Fp, Fm, Fx, Fy, Fv_x, Fv_y, Nx, Ny, NG, dξdx, dξdy, dηdx, dηdy, J, dt1)

        @cuda threads=nthreads blocks=nblock linComb(U, Un, Nx, Ny, NG, Ncons, 2/3, 1/3)
        @cuda threads=nthreads blocks=nblock linComb(ρi, ρn, Nx, Ny, NG, Nspecs, 2/3, 1/3)
        @cuda threads=nthreads blocks=nblock fillSpec(ρi, U, NG, Nx, Ny)
        @cuda threads=nthreads blocks=nblock fillGhost(U, NG, Nx, Ny)
        @cuda threads=nthreads blocks=nblock correction(U, ρi, NG, Nx, Ny)

        # Reaction Step
        @cuda threads=nthreads blocks=nblock c2Prim(U, Q, Nx, Ny, NG, 1.4, 287)
        @cuda threads=nthreads blocks=nblock pre_input(inputs, inputs_norm, Q, ρi, lambda, inputs_mean, inputs_std, Nx, Ny, NG)
        yt_pred = model(cu(inputs_norm), ps, st)[1]
        @. yt_pred = yt_pred * labels_std + labels_mean
        @cuda threads=nthreads blocks=nblock post_predict(yt_pred, inputs, U, Q, ρi, dt2, lambda, Nx, Ny, NG)
        @cuda threads=nthreads blocks=nblock fillSpec(ρi, U, NG, Nx, Ny)
        @cuda threads=nthreads blocks=nblock fillGhost(U, NG, Nx, Ny)
        @cuda threads=nthreads blocks=nblock correction(U, ρi, NG, Nx, Ny)
    end
    return
end
