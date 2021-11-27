"""
    AbstractGate

Abstract supertype for all quantum gates.
"""
abstract type AbstractGate end

"""
    HaarGate <: AbstractGate

Random unitary gate.
"""
struct HaarGate{N,M} <: AbstractGate
    idcs::NTuple{M,Int}

    lb::SpinBasis{1//2,Int64}
    gb::CompositeBasis{Vector{Int},NTuple{SpinBasis{1//2,Int64}}}

    H::Matrix{ComplexF64}

    dt::Float64
    T::Float64
end

"""
    HaarGate((i_1,...,i_M), N)

Construct a random `HaarGate` on the `i_1`th to `i_M`-th qubits of a `N`-qubit circuit.
"""
function HaarGate(idcs::NTuple{M,Int}, N::Int64; dt=missing) where M
    @assert length(idcs) <= N
    lb = SpinBasis(1//2)
    gb = SpinBasis(1//2)^N
    T = spectral_width(length(g.lb)^length(idcs))
    dt = ismissing(dt) ? 5e-2T : dt

    U = rand(CUE(length(g.lb)^length(idcs)))
    H = im*log(U) / g.T

    HaarGate(idcs, lb, gb, H, dt, T)
end

spectral_width(m) = 3.1611209037341705 * m^(-0.4167529049808372) * log(0.24110347945022853 * m) + 4.485346312064226

"""
    hamiltonian(g::HaarGate)

Generate the hamiltonian of a `HaarGate`.
"""
hamiltonian(g::HaarGate) = g.H

"""
    RowGate <: AbstractGate

Row of gates.
"""
struct RowGate{N} <: AbstractGate
    lb::SpinBasis{1//2,Int64}
    gb::CompositeBasis{Vector{Int},NTuple{SpinBasis{1//2,Int64}}}

    H::Matrix{ComplexF64}

    dt::Float64
    T::Float64
end

function RowGate(gs::AbstractArray{G}) where G <: AbstractGate
    @assert all(g->g.lb==gs[1].lb, gs)
    @assert all(g->g.gb==gs[1].gb, gs)
    @assert all(g->g.T ==gs[1].T , gs)

    H = mapreduce(hamiltonian, +, gs)
    dt = minimum(map(g->g.dt, gs))

    RowGate(lb, gb, H, dt, T)
end

"""
    hamiltonian(g::RowGate)

Generate the hamiltonian of a `RowGate`.
"""
hamiltonian(g::RowGate) = g.H

"""
    generate_random_circuit(N)

Generate a vector of quantum gates corresponding to the QFT algorithm for `N` qubits.
"""
function generate_random_circuit(N::Int64, M::Int, depth::Int; pbc=true)
    @assert M <= N
    @assert M == 2 "Arbitrary M not yet implemented"

    map(1:depth) do d
        n = isodd(d) ? 0 : 1
        ngates = div(N-Int(!pbc)*n,2)
        RowGate([HaarGate((2i-1+n,mod1(2i+n,N)),N) for i in 1:ngates])
    end
end
