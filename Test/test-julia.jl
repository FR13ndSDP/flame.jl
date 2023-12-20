using BenchmarkTools

# initialize
const N::Int64 = 1024
ϕn = ones(Float64, N*N)
ϕ = ones(Float64, N*N)

function test(ϕ, ϕn, N)
    for i ∈ 2:N-1, j ∈ 2:N-1
        @inbounds ϕn[(i-1)*N+j] = 0.25*(ϕ[(i-2)*N+j]+ϕ[i*N+j]+ϕ[(i-1)*N+j-1]+ϕ[(i-1)*N+j+1])
    end
end

@time test(ϕ, ϕn, N)
@benchmark for n=1:1000
    test(ϕ, ϕn, N)
end