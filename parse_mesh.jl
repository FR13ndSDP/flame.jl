# Mesh for flat plate
using JLD2
using WriteVTK

const NG::Int64 = 4
const Nx::Int64 = 1024
const Ny::Int64 = 256
const Lx::Float64 = 1
const Ly::Float64 = 0.1
const Nx_tot::Int64 = Nx + 2*NG
const Ny_tot::Int64 = Ny + 2*NG

x = zeros(Float64, Nx_tot, Ny_tot)
y = zeros(Float64, Nx_tot, Ny_tot)

for j ∈ 1:Ny, i ∈ 1:Nx
    x[i+NG, j+NG] = (i-1) * (Lx/(Nx-1))
    y[i+NG, j+NG] = Ly * (0.75*((j-1)/(Ny-1))^3 + 0.25*(j-1)/(Ny-1))
end

# get ghost location
for j ∈ NG+1:Ny+NG, i ∈ 1:NG
    x[i, j] = 2*x[NG+1, j] - x[2*NG+2-i, j]
    y[i, j] = 2*y[NG+1, j] - y[2*NG+2-i, j]
end

for j ∈ NG+1:Ny+NG, i ∈ Nx+NG+1:Nx+2*NG
    x[i, j] = 2*x[Nx+NG, j] - x[2*NG+2*Nx-i, j]
    y[i, j] = 2*y[Nx+NG, j] - y[2*NG+2*Nx-i, j]
end

for j ∈ 1:NG, i ∈ NG+1:Nx+NG
    x[i, j] = 2*x[i, NG+1] - x[i, 2*NG+2-j]
    y[i, j] = 2*y[i, NG+1] - y[i, 2*NG+2-j]
end

for j ∈ Ny+NG+1:Ny+2*NG, i ∈ NG+1:Nx+NG
    x[i, j] = 2*x[i, Ny+NG] - x[i, 2*NG+2*Ny-j]
    y[i, j] = 2*y[i, Ny+NG] - y[i, 2*NG+2*Ny-j]
end

#corner ghost
for j ∈ Ny+NG+1:Ny+2*NG, i ∈ 1:NG
    x[i, j] = x[i, Ny+NG] + x[NG+1, j] - x[NG+1, Ny+NG]
    y[i, j] = y[i, Ny+NG] + y[NG+1, j] - y[NG+1, Ny+NG]
end

for j ∈ 1:NG, i ∈ 1:NG
    x[i, j] = x[i, NG+1] + x[NG+1, j] - x[NG+1, NG+1]
    y[i, j] = y[i, NG+1] + y[NG+1, j] - y[NG+1, NG+1]
end

for j ∈ Ny+NG+1:Ny+2*NG, i ∈ Nx+NG+1:Nx+2*NG
    x[i, j] = x[i, Ny+NG] + x[Nx+NG, j] - x[Nx+NG, Ny+NG]
    y[i, j] = y[i, Ny+NG] + y[Nx+NG, j] - y[Nx+NG, Ny+NG]
end

for j ∈ 1:NG, i ∈ Nx+NG+1:Nx+2*NG
    x[i, j] = x[i, NG+1] + x[Nx+NG, j] - x[Nx+NG, NG+1]
    y[i, j] = y[i, NG+1] + y[Nx+NG, j] - y[Nx+NG, NG+1]
end

# compute jacobian
function CD6(f)
    fₓ = 1/60*(f[7]-f[1]) - 3/20*(f[6]-f[2]) + 3/4*(f[5]-f[3])
    return fₓ
end

function CD2_L(f)
    fₓ = 2*f[2] - 0.5*f[3] - 1.5*f[1]
    return fₓ
end

function CD2_R(f)
    fₓ = -2*f[2] + 0.5*f[1] + 1.5*f[3]
    return fₓ
end

# Jacobians
dxdξ = zeros(Float64, Nx_tot, Ny_tot)
dxdη = zeros(Float64, Nx_tot, Ny_tot)
dydξ = zeros(Float64, Nx_tot, Ny_tot)
dydη = zeros(Float64, Nx_tot, Ny_tot)

dξdx = zeros(Float64, Nx_tot, Ny_tot)
dηdx = zeros(Float64, Nx_tot, Ny_tot)
dξdy = zeros(Float64, Nx_tot, Ny_tot)
dηdy = zeros(Float64, Nx_tot, Ny_tot)
J  = zeros(Float64, Nx_tot, Ny_tot)

for j ∈ 1:Ny_tot, i ∈ 4:Nx_tot-3
    dxdξ[i, j] = CD6(x[i-3:i+3, j])
    dydξ[i, j] = CD6(y[i-3:i+3, j])
end

for j ∈ 4:Ny_tot-3, i ∈ 1:Nx_tot
    dxdη[i, j] = CD6(x[i, j-3:j+3])
    dydη[i, j] = CD6(y[i, j-3:j+3])
end

# boundary
for j ∈ 1:Ny_tot, i ∈ 1:3
    dxdξ[i, j] = CD2_L(x[i:i+2, j])
    dydξ[i, j] = CD2_L(y[i:i+2, j])
end

for j ∈ 1:Ny_tot, i ∈ Nx_tot-2:Nx_tot
    dxdξ[i, j] = CD2_R(x[i-2:i, j])
    dydξ[i, j] = CD2_R(y[i-2:i, j])
end

for j ∈ 1:3, i ∈ 1:Nx_tot
    dxdη[i, j] = CD2_L(x[i, j:j+2])
    dydη[i, j] = CD2_L(y[i, j:j+2])
end

for j ∈ Ny-2:Ny+2*NG, i ∈ 1:Nx+2*NG
    dxdη[i, j] = CD2_R(x[i, j-2:j])
    dydη[i, j] = CD2_R(y[i, j-2:j])
end

@. J = 1 / (dxdξ*dydη - dxdη*dydξ)

# actually after * J⁻
dξdx = dydη
dξdy = -dxdη
dηdx = -dydξ
dηdy = dxdξ

@save "metrics.jld2" NG Nx Ny dξdx dξdy dηdx dηdy J x y

vtk_grid("mesh.vts", x, y) do vtk
end