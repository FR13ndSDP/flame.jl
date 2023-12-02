# Mesh for wedge with ghost cell
using JLD2
using WriteVTK

global NG::Int64 = 2
global Nx::Int64 = 512
global Ny::Int64 = 80
global Lx_flat::Float64 = 1.0
global Lx_corner::Float64 = 0.5
global y1::Float64 = 0.0001
global θ::Float64 = 24/180*π

x = zeros(Float64, Nx, Ny)
y = zeros(Float64, Nx, Ny)

midpoint = cld(Nx, 2)

for i = 1:midpoint
    for j = 1:Ny
        x[i, j] = (i-1) * (Lx_flat/(midpoint-1)) - Lx_flat
        y[i, j] = j*(j-1)/2 * y1
    end
end

for i = midpoint+1:Nx
    for j = 1:Ny
        x[i, j] = (i - midpoint) * (Lx_corner/(Nx-midpoint)) * cos(θ)
        y[i, j] = (i - midpoint) * (Lx_corner/(Nx-midpoint)) * sin(θ) + j*(j-1)/2*y1
    end
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
dxdξ = zeros(Float64, Nx, Ny)
dxdη = zeros(Float64, Nx, Ny)
dydξ = zeros(Float64, Nx, Ny)
dydη = zeros(Float64, Nx, Ny)

dξdx = zeros(Float64, Nx, Ny)
dηdx = zeros(Float64, Nx, Ny)
dξdy = zeros(Float64, Nx, Ny)
dηdy = zeros(Float64, Nx, Ny)
J  = zeros(Float64, Nx, Ny)

for i = 4:Nx-3
    for j = 1:Ny
        dxdξ[i, j] = CD6(x[i-3:i+3, j])
        dydξ[i, j] = CD6(y[i-3:i+3, j])
    end
end

for i = 1:Nx
    for j = 4:Ny-3
        dxdη[i, j] = CD6(x[i, j-3:j+3])
        dydη[i, j] = CD6(y[i, j-3:j+3])
    end
end

# boundary
for i = 1:3
    for j = 1:Ny
        dxdξ[i, j] = CD2_L(x[i:i+2, j])
        dydξ[i, j] = CD2_L(y[i:i+2, j])
    end
end

for i = Nx-2:Nx
    for j = 1:Ny
        dxdξ[i, j] = CD2_R(x[i-2:i, j])
        dydξ[i, j] = CD2_R(y[i-2:i, j])
    end
end

for i = 1:Nx
    for j = 1:3
        dxdη[i, j] = CD2_L(x[i, j:j+2])
        dydη[i, j] = CD2_L(y[i, j:j+2])
    end
end

for i = 1:Nx
    for j = Ny-2:Ny
        dxdη[i, j] = CD2_R(x[i, j-2:j])
        dydη[i, j] = CD2_R(y[i, j-2:j])
    end
end

J = 1 ./ (dxdξ.*dydη - dxdη.*dydξ)

# actually after * J⁻
dξdx = dydη
dξdy = -dxdη
dηdx = -dydξ
dηdy = dxdξ

Nx -= 2*NG
Ny -= 2*NG
@save "metrics.jld2" NG Nx Ny dξdx dξdy dηdx dηdy J x y

vtk_grid("mesh.vts", x, y) do vtk
end