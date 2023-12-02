using Plots
using JSON, PyCall
using Lux, JLD2

@load "luxmodel.jld2" model ps st

# Validation
# Call Cantera
mech = "./drm19.yaml"
ct = pyimport("cantera")
ct_gas = ct.Solution(mech)
ct_gas.TPX = 1600, 1.0*ct.one_atm, "CH4:0.5 O2:1 N2:3.76"
r = ct.IdealGasConstPressureReactor(ct_gas, name="R1")
sim = ct.ReactorNet([r])
T_evo_ct = zeros(Float64, 1000)
Y_evo_ct = zeros(Float64, (20, 1000))
T_evo_ct[1] = ct_gas.T
Y_evo_ct[:, 1] = ct_gas.Y

@time for i=1:999
    sim.advance(i*1e-6)
    T_evo_ct[i+1] = ct_gas.T
    Y_evo_ct[:, i+1] = ct_gas.Y
end

# Lux.jl
gas = ct.Solution(mech)
gas.TPX = 1600, 1.0*ct.one_atm, "CH4:0.5 O2:1 N2:3.76"
T_evo = zeros(Float64, 1000)
Y_evo = zeros(Float64, (20, 1000))
T_evo[1] = gas.T
Y_evo[:, 1] = gas.Y

input = zeros(Float64, 21)
j = JSON.parsefile("norm-new.json")
dt = j["dt"]
lambda = j["lambda"]
inputs_mean = convert(Vector{Float64}, j["inputs_mean"])
inputs_std = convert(Vector{Float64}, j["inputs_std"])
labels_mean = convert(Vector{Float64}, j["labels_mean"])
labels_std = convert(Vector{Float64}, j["labels_std"])

@time for i=1:999
    input[1] = gas.T
    input[2] = gas.P
    input[3:end] = (gas.Y[1:19].^lambda .- 1) ./ lambda
    input_norm = (input .- inputs_mean) ./ inputs_std
    y_pred = Lux.apply(model, input_norm, ps, st)[1]
    y_pred = y_pred .* labels_std .+ labels_mean
    y_pred = (lambda .* (y_pred .* dt .+ input[3:end]) .+ 1).^(1/lambda)
    append!(y_pred, gas.Y[end])
    gas.HPY = gas.h, gas.P, y_pred
    T_evo[i+1] = gas.T
    Y_evo[:, i+1] = gas.Y
end

# compute relative error
Err = abs.(Y_evo .- Y_evo_ct)./(Y_evo .+ 1f-20)
max_err = [maximum(c) for c in eachslice(Err, dims=1)]
println("Max relative error for Y: $max_err")

# Plot
# with_theme(theme_web()) do
# GLMakie.activate!()
# fig = Figure(resolution=(1200, 600))
# ax1 = Axis(fig[1, 1]; xlabel="x", ylabel="y")

# l = lines!(ax1, T_evo_ct; linewidth=3)
# s = lines!(ax1, T_evo; linewidth=3)

# axislegend(ax1, [l, s], ["cantera", "Lux.jl"])

# ax2 = Axis(fig[1, 2]; xlabel="x", ylabel="y")

# l = lines!(ax2, Y_evo_ct[1, :]; linewidth=3)
# s = lines!(ax2, Y_evo[1, :]; linewidth=3)

# axislegend(ax2, [l, s], ["cantera", "Lux.jl"])

# fig
gr()
p1 = plot([T_evo T_evo_ct], w = 3, lab = ["predict" "cantera"], ls=[:dot :solid], lw = 2)
p2 = plot([Y_evo[1:19, :]' Y_evo_ct[1:19, :]'], ls=[:dot :solid], lw = 2)

plot(p1, p2, layout=(1,2), legend=false, size=(800,400))
# end
