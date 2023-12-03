# Mesh for wedge with ghost cell
using JLD2
using WriteVTK

global NG::Int64 = 4
global Nx::Int64 = 512 #include ghost cell
global Ny::Int64 = 256 #include ghost cell
global Lx_flat::Float64 = 1.0
global Lx_corner::Float64 = 0.5
global y1::Float64 = 0.00001
global θ::Float64 = 24/180*π
global Nx_tot::Int64 = Nx + 2*NG
global Ny_tot::Int64 = Ny + 2*NG

x = zeros(Float64, Nx_tot, Ny_tot)
y = zeros(Float64, Nx_tot, Ny_tot)

midpoint = cld(Nx, 2)

for i = 1:midpoint
    for j = 1:Ny
        x[i+NG, j+NG] = (i-1) * (Lx_flat/(midpoint-1)) - Lx_flat
        y[i+NG, j+NG] = j*(j-1)/2 * y1
    end
end

for i = midpoint+1:Nx
    for j = 1:Ny
        x[i+NG, j+NG] = (i - midpoint) * (Lx_corner/(Nx-midpoint)) * cos(θ)
        y[i+NG, j+NG] = (i - midpoint) * (Lx_corner/(Nx-midpoint)) * sin(θ) + j*(j-1)/2*y1
    end
end

# get ghost location
for i = 1:NG
    for j = NG+1:Ny+NG
        x[i, j] = 2*x[NG+1, j] - x[2*NG+2-i, j]
        y[i, j] = 2*y[NG+1, j] - y[2*NG+2-i, j]
    end
end

for i = Nx+NG+1:Nx+2*NG
    for j = NG+1:Ny+NG
        x[i, j] = 2*x[Nx+NG, j] - x[2*NG+2*Nx-i, j]
        y[i, j] = 2*y[Nx+NG, j] - y[2*NG+2*Nx-i, j]
    end
end

for i = NG+1:Nx+NG
    for j = 1:NG
        x[i, j] = 2*x[i, NG+1] - x[i, 2*NG+2-j]
        y[i, j] = 2*y[i, NG+1] - y[i, 2*NG+2-j]
    end
end

for i = NG+1:Nx+NG
    for j = Ny+NG+1:Ny+2*NG
        x[i, j] = 2*x[i, Ny+NG] - x[i, 2*NG+2*Ny-j]
        y[i, j] = 2*y[i, Ny+NG] - y[i, 2*NG+2*Ny-j]
    end
end

#corner ghost
for i = 1:NG
    for j = Ny+NG+1:Ny+2*NG
        x[i, j] = x[i, Ny+NG] + x[NG+1, j] - x[NG+1, Ny+NG]
        y[i, j] = y[i, Ny+NG] + y[NG+1, j] - y[NG+1, Ny+NG]
    end
end

for i = 1:NG
    for j = 1:NG
        x[i, j] = x[i, NG+1] + x[NG+1, j] - x[NG+1, NG+1]
        y[i, j] = y[i, NG+1] + y[NG+1, j] - y[NG+1, NG+1]
    end
end

for i = Nx+NG+1:Nx+2*NG
    for j = Ny+NG+1:Ny+2*NG
        x[i, j] = x[i, Ny+NG] + x[Nx+NG, j] - x[Nx+NG, Ny+NG]
        y[i, j] = y[i, Ny+NG] + y[Nx+NG, j] - y[Nx+NG, Ny+NG]
    end
end

for i = Nx+NG+1:Nx+2*NG
    for j = 1:NG
        x[i, j] = x[i, NG+1] + x[Nx+NG, j] - x[Nx+NG, NG+1]
        y[i, j] = y[i, NG+1] + y[Nx+NG, j] - y[Nx+NG, NG+1]
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
dxdξ = zeros(Float64, Nx_tot, Ny_tot)
dxdη = zeros(Float64, Nx_tot, Ny_tot)
dydξ = zeros(Float64, Nx_tot, Ny_tot)
dydη = zeros(Float64, Nx_tot, Ny_tot)

dξdx = zeros(Float64, Nx_tot, Ny_tot)
dηdx = zeros(Float64, Nx_tot, Ny_tot)
dξdy = zeros(Float64, Nx_tot, Ny_tot)
dηdy = zeros(Float64, Nx_tot, Ny_tot)
J  = zeros(Float64, Nx_tot, Ny_tot)

for i = 4:Nx_tot-3
    for j = 1:Ny_tot
        dxdξ[i, j] = CD6(x[i-3:i+3, j])
        dydξ[i, j] = CD6(y[i-3:i+3, j])
    end
end

for i = 1:Nx_tot
    for j = 4:Ny_tot-3
        dxdη[i, j] = CD6(x[i, j-3:j+3])
        dydη[i, j] = CD6(y[i, j-3:j+3])
    end
end

# boundary
for i = 1:3
    for j = 1:Ny_tot
        dxdξ[i, j] = CD2_L(x[i:i+2, j])
        dydξ[i, j] = CD2_L(y[i:i+2, j])
    end
end

for i = Nx_tot-2:Nx_tot
    for j = 1:Ny_tot
        dxdξ[i, j] = CD2_R(x[i-2:i, j])
        dydξ[i, j] = CD2_R(y[i-2:i, j])
    end
end

for i = 1:Nx_tot
    for j = 1:3
        dxdη[i, j] = CD2_L(x[i, j:j+2])
        dydη[i, j] = CD2_L(y[i, j:j+2])
    end
end

for i = 1:Nx+2*NG
    for j = Ny-2:Ny+2*NG
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

@save "metrics.jld2" NG Nx Ny dξdx dξdy dηdx dηdy J x y

vtk_grid("mesh.vts", x, y) do vtk
end