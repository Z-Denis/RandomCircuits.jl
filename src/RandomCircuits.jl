module RandomCircuits

using QuantumOpticsBase, QuantumInformation
using LinearAlgebra, SparseArrays, ElasticArrays
using OrdinaryDiffEq, DiffEqBase, DiffEqCallbacks

include("types.jl")
export AbstractGate, HaarGate, hamiltonian, RowGate, generate_random_circuit

include("timeevolution.jl")
export compute_tstops, time_evolution

end
