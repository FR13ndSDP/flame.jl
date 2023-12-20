#=  cartesian rank
    i(0)---------------
    |   0   1   2   3  |
    |   4   5   6   7  |
    |   8   9  10  11  |
    |  12  13  14  15  |
    ---------------- j(1)
=#

using MPI
using HDF5
using StaticArrays
MPI.Init()

const Nx::Int64 = 256
const Ny::Int64 = 256

const Nproc_x::Int64 = 2
const Nproc_y::Int64 = 2
const Nghost::Int64 = 1
const root::Int64 = 0
const Iperiodic = (false, false)

function laplacian(ϕ, ϕn, lo, hi)
    for j ∈ lo[2]:hi[2], i ∈ lo[1]:hi[1]
        @inbounds ϕn[i,j] = 0.25*(ϕ[i-1,j]+ϕ[i+1,j]+ϕ[i,j-1]+ϕ[i,j+1])
    end
end

function fill_boundary(ϕ, rank, Nproc_x)
    if rank ÷ Nproc_x == 0
        ϕ[1, :] .= 10.0
    end
end

function exchange_data(ϕ, ϕn)
    ϕ .= ϕn
end

function exchange_ghost(ϕ, comm_cart, comm, lo, hi, lo_g, hi_g)
    # exchange boundary
    src, dst = MPI.Cart_shift(comm_cart, 0, 1)
    sendbuf = @view ϕ[hi[1]-Nghost+1:hi[1], :]
    recvbuf = @view ϕ[lo_g[1]:lo[1]-1, :]
    status = MPI.Sendrecv!(sendbuf, recvbuf, comm; dest=dst, source=src)

    src, dst = MPI.Cart_shift(comm_cart, 0, -1)
    sendbuf = @view ϕ[lo[1]:lo[1]+Nghost-1, :]
    recvbuf = @view ϕ[hi[1]+1:hi_g[1], :]
    status = MPI.Sendrecv!(sendbuf, recvbuf, comm; dest=dst, source=src)

    src, dst = MPI.Cart_shift(comm_cart, 1, 1)
    sendbuf = @view ϕ[:, hi[2]-Nghost+1:hi[2]]
    recvbuf = @view ϕ[:, lo_g[2]:lo[2]-1]
    status = MPI.Sendrecv!(sendbuf, recvbuf, comm; dest=dst, source=src)

    src, dst = MPI.Cart_shift(comm_cart, 1, -1)
    sendbuf = @view ϕ[:, lo[2]:lo[2]+Nghost-1]
    recvbuf = @view ϕ[:, hi[2]+1:hi_g[2]]
    status = MPI.Sendrecv!(sendbuf, recvbuf, comm; dest=dst, source=src)
end

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
Nprocs = MPI.Comm_size(comm)

if Nprocs != Nproc_x * Nproc_y
    error("Not correct processes")
end
comm_cart = MPI.Cart_create(comm, [Nproc_x, Nproc_y]; periodic=Iperiodic)

# calculate local indices w/o ghost cells
lo = [1+Nghost,1+Nghost]
hi = [Nx ÷ Nproc_x+Nghost, Ny ÷ Nproc_y+Nghost]
lo_g = [1, 1]
hi_g = [hi[1]+Nghost, hi[2]+Nghost]

# initialize
ϕn = ones(Float64, hi_g[1], hi_g[2])
ϕ = ones(Float64, hi_g[1], hi_g[2])
# fill boundary
fill_boundary(ϕ, rank, Nproc_x)

laplacian(ϕ, ϕn, lo, hi)

exchange_data(ϕ, ϕn)

exchange_ghost(ϕ, comm_cart, comm, lo, hi, lo_g, hi_g)

MPI.Barrier(comm)

fill_boundary(ϕ, rank, Nproc_x)

@time for n = 1:50000
    laplacian(ϕ, ϕn, lo, hi)

    exchange_data(ϕ, ϕn)

    exchange_ghost(ϕ, comm_cart, comm, lo, hi, lo_g, hi_g)

    MPI.Barrier(comm)

    fill_boundary(ϕ, rank, Nproc_x)
end

ϕng = @view ϕ[lo[1]:hi[1], lo[2]:hi[2]] # remove ghost
h5open("test.h5", "w", comm) do f
    dset = create_dataset(
        f,
        "/phi",
        datatype(Float64),
        dataspace(hi[1]-Nghost, hi[2]-Nghost, Nprocs);
        chunk=(hi[1]-Nghost, hi[2]-Nghost, 1),
        dxpl_mpio=:collective
    )
    dset[:, :, rank + 1] = ϕng
end

MPI.Finalize()


