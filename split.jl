#For N-S, range: 1->N+2*NG
function fluxSplit(Q, U, Fp, Fm, Nx, Ny, NG, Ax, Ay)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    if i > Nx+2*NG || j > Ny+2*NG
        return
    end

    @inbounds ρ = Q[i, j, 1]
    @inbounds u = Q[i, j, 2]
    @inbounds v = Q[i, j, 3]
    @inbounds p = Q[i, j, 4]
    @inbounds c = Q[i, j, 6]
    @inbounds ei = U[i, j, 4] - 0.5 * ρ * (u^2 + v^2)
    @inbounds A1 = Ax[i, j]
    @inbounds A2 = Ay[i, j]

    γ = p/ei + 1

    ss = CUDA.sqrt(A1*A1 + A2*A2)
    E1 = A1*u + A2*v
    E2 = E1 - c*ss
    E3 = E1 + c*ss

    ss = 1/ss

    A1 *= ss
    A2 *= ss

    E1P = (E1 + CUDA.abs(E1)) * 0.5
    E2P = (E2 + CUDA.abs(E2)) * 0.5
    E3P = (E3 + CUDA.abs(E3)) * 0.5

    E1M = E1 - E1P
    E2M = E2 - E2P
    E3M = E3 - E3P

    uc1 = u - c * A1
    uc2 = u + c * A1
    vc1 = v - c * A2
    vc2 = v + c * A2

    vvc1 = (uc1*uc1 + vc1*vc1) * 0.50
    vvc2 = (uc2*uc2 + vc2*vc2) * 0.50
    vv = (γ - 1.0) * (u*u + v*v)
    W2 = (3-γ)/(2*(γ-1)) * c * c

    tmp1 = ρ/(2 * γ)
    tmp2 = 2 * (γ - 1)
    @inbounds Fp[i, j, 1] = tmp1 * (tmp2 * E1P + E2P + E3P);
    @inbounds Fp[i, j, 2] = tmp1 * (tmp2 * E1P * u + E2P * uc1 + E3P * uc2)
    @inbounds Fp[i, j, 3] = tmp1 * (tmp2 * E1P * v + E2P * vc1 + E3P * vc2)
    @inbounds Fp[i, j, 4] = tmp1 * (E1P * vv + E2P * vvc1 + E3P * vvc2 + W2 * (E2P + E3P))

    @inbounds Fm[i, j, 1] = tmp1 * (tmp2 * E1M + E2M + E3M);
    @inbounds Fm[i, j, 2] = tmp1 * (tmp2 * E1M * u + E2M * uc1 + E3M * uc2);
    @inbounds Fm[i, j, 3] = tmp1 * (tmp2 * E1M * v + E2M * vc1 + E3M * vc2);
    @inbounds Fm[i, j, 4] = tmp1 * (E1M * vv + E2M * vvc1 + E3M * vvc2 + W2 * (E2M + E3M));
    return 
end

# For species, range 1->N+2*NG
function split(ρi, Q, U, Fp, Fm, Ax, Ay, Nx, Ny, NG)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    if i > Nx+2*NG || j > Ny+2*NG
        return
    end

    ρ = Q[i, j, 1]
    u = Q[i, j, 2]
    v = Q[i, j, 3]
    p = Q[i, j, 4]
    c = Q[i, j, 6]
    ei = U[i, j, 4] - 0.5 * ρ * (u^2 + v^2)
    A1 = Ax[i, j]
    A2 = Ay[i, j]

    γ = p/ei + 1

    ss = CUDA.sqrt(A1*A1 + A2*A2)
    E1 = A1*u + A2*v
    E2 = E1 - c*ss
    E3 = E1 + c*ss

    ss = 1/ss

    A1 *= ss
    A2 *= ss

    E1P = (E1 + CUDA.abs(E1)) * 0.5
    E2P = (E2 + CUDA.abs(E2)) * 0.5
    E3P = (E3 + CUDA.abs(E3)) * 0.5

    E1M = E1 - E1P
    E2M = E2 - E2P
    E3M = E3 - E3P

    tmp1 = 1/(2 * γ)
    tmp2 = 2 * (γ - 1)

    for n = 1:Nspecs
        Fp[i, j, n] = tmp1 * (tmp2 * E1P + E2P + E3P) * ρi[i, j, n]
        Fm[i, j, n] = tmp1 * (tmp2 * E1M + E2M + E3M) * ρi[i, j, n]
    end
    return
end