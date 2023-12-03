include("solver.jl")
using BenchmarkTools

@load "metrics.jld2" NG Nx Ny dξdx dξdy dηdx dηdy J x y

dt::Float64 = 4e-8
Time::Float64 = 3e-3

U = zeros(Float64, Nx+2*NG, Ny+2*NG, 4)

#initialization on CPU
# Mach 6 inlet
T = 79
P = 0.012 * 101325
c = sqrt(1.4 * 287 * T)
u = 6*c
ρ = P/(287 * T)

U[:, :, 1] .= ρ
U[:, :, 2] .= ρ * u
U[:, :, 3] .= 0.0
U[:, :, 4] .= P/(1.4-1) + 0.5*ρ*u^2


U_gpu = CuArray(U)
dξdx_d = CuArray(dξdx)
dξdy_d = CuArray(dξdy)
dηdx_d = CuArray(dηdx)
dηdy_d = CuArray(dηdy)
J_d = CuArray(J)

time_step!(U_gpu, Time, dt, dξdx_d, dξdy_d, dηdx_d, dηdy_d, J_d, Nx, Ny, NG)
copyto!(U, U_gpu)

# remove ghost cells and write 
rho = U[NG+1:Nx+NG, NG+1:Ny+NG, 1]
u = U[NG+1:Nx+NG, NG+1:Ny+NG, 2]./rho
v = U[NG+1:Nx+NG, NG+1:Ny+NG, 3]./rho
p = (U[NG+1:Nx+NG, NG+1:Ny+NG, 4] - 0.5.*rho.*(u.^2+v.^2)) * 0.4
T = p./(287.0 .* rho)
x = x[NG+1:Nx+NG, NG+1:Ny+NG]
y = y[NG+1:Nx+NG, NG+1:Ny+NG]
vtk_grid("result.vts", x, y) do vtk
    vtk["rho"] = rho
    vtk["u"] = u
    vtk["v"] = v
    vtk["p"] = p
    vtk["T"] = T
end 