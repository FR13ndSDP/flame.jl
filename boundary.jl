function fillGhost(U, NG, Nx, Ny)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    
    if i > Nx+2*NG || j > Ny+2*NG
        return
    end

    gamma::Float64 = 1.4
    T_wall::Float64 = 5323
    Rg::Float64 = 287

    noise::Float64 = rand() * 0 * 3757.810529345963
    # Mach 10 inlet
    if i <= 10
        @inbounds U[i, j, 1] = 0.035651512619407424
        @inbounds U[i, j, 2] = 0.035651512619407424 * 3757.810529345963
        @inbounds U[i, j, 3] = 0.0
        @inbounds U[i, j, 4] = 3596/0.4 + 0.5*0.035651512619407424*3757.810529345963^2
    elseif i > Nx + NG -1
        for n = 1:4
            @inbounds U[i, j, n] = U[Nx+NG-1, j, n]
        end
    else
        if j == NG+1 
            @inbounds U[i, j, 2] = 0
            if i <= 40 && i >= 30
                @inbounds U[i, j, 3] = noise * U[i, j+1, 1]
            else
                @inbounds U[i, j, 3] = 0
            end
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

function fillSpec(ρi, U, NG, Nx, Ny)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    
    if i > Nx+2*NG || j > Ny+2*NG
        return
    end

    if i <= 10
        for n = 1:Nspecs
            ρi[i, j, n] = 0
        end
        ρi[i, j, 2] = 0.035651512619407424 * 0.233
        ρi[i, j, 7] = 0.035651512619407424 * 0.767
    elseif i > Nx + NG -1
        for n = 1:Nspecs
            ρi[i, j, n] = ρi[Nx+NG-1, j, n]
        end
    else
        if j <= NG+1
            for n = 1:Nspecs
                ρi[i, j, n] = ρi[i, NG+2, n]
            end
        elseif j > Ny+NG-1
            for n = 1:Nspecs
                ρi[i, j, n] = ρi[i, Ny+NG-1, n]
            end
        end
    end
    return
end