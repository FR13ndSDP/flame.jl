include("solver.jl")
using PyCall
import Adapt

# load mesh metrics
@load "metrics.jld2" NG Nx Ny dξdx dξdy dηdx dηdy J x y

# global variables, do not change name
const dt::Float64 = 1e-8
const Time::Float64 = 1e-4
const Nspecs::Int64 = 8 # number of species
const Ncons::Int64 = 4 # ρ ρu ρv E 
const Nprim::Int64 = 6 # ρ u v p T c 
const mech = "./NN/air.yaml"

struct thermoProperty{IT, RT, VT, MT, TT}
    Nspecs::IT
    Ru::RT
    mw::VT
    coeffs_sep::VT
    coeffs_lo::MT
    coeffs_hi::MT
    visc_poly::MT
    conduct_poly::MT
    binarydiff_poly::TT
end

Adapt.@adapt_structure thermoProperty

U = zeros(Float64, Nx+2*NG, Ny+2*NG, Ncons)
ρi = zeros(Float64, Nx+2*NG, Ny+2*NG, Nspecs)

#initialization on CPU
function initialize(U, mech)
    ct = pyimport("cantera")
    gas = ct.Solution(mech)
    T::Float64 = 350
    P::Float64 = 3596
    gas.TPY = T, P, "N2:0.767 O2:0.233"
    ρ::Float64 = gas.density
    c::Float64 = sqrt(1.4 * P / ρ)
    u::Float64 = 10 * c

    U[:, :, 1] .= ρ
    U[:, :, 2] .= ρ * u
    U[:, :, 3] .= 0.0
    U[:, :, 4] .= P/(1.4-1) + 0.5 * ρ * u^2
    for j ∈ 1:Ny+2*NG, i ∈ 1:Nx+2*NG
        ρi[i, j, :] .= gas.Y .* ρ
    end
end

thermo = initThermo(mech, Nspecs)
initialize(U, mech)

U_d = CuArray(U)
ρi_d = CuArray(ρi)
dξdx_d = CuArray(dξdx)
dξdy_d = CuArray(dξdy)
dηdx_d = CuArray(dηdx)
dηdy_d = CuArray(dηdy)
J_d = CuArray(J)

@time time_step(U_d, ρi_d, dξdx_d, dξdy_d, dηdx_d, dηdy_d, J_d, Nx, Ny, NG, dt)
copyto!(U, U_d)
copyto!(ρi, ρi_d)

rho = U[:, :, 1]
u =   U[:, :, 2]./rho
v =   U[:, :, 3]./rho
p = @. (U[:, :, 4] - 0.5*rho*(u^2+v^2)) * 0.4
T = @. p/(287.0 * rho)

O =   ρi[:, :, 1]
O2 =  ρi[:, :, 2]
N =   ρi[:, :, 3]
NO =  ρi[:, :, 4]
NO2 = ρi[:, :, 5]
N2O = ρi[:, :, 6]
N2 =  ρi[:, :, 7]
AR =  ρi[:, :, 8]
vtk_grid("result.vts", x, y) do vtk
    vtk["rho"] = rho
    vtk["u"] = u
    vtk["v"] = v
    vtk["p"] = p
    vtk["T"] = T
    vtk["O"] = O
    vtk["O2"] = O2
    vtk["N"] = N
    vtk["NO"] = NO
    vtk["NO2"] = NO2
    vtk["N2O"] = N2O
    vtk["N2"] = N2
    vtk["AR"] = AR
end 
