module RandomCircuits

using QuantumOpticsBase, QuantumInformation
using LinearAlgebra, SparseArrays, ElasticArrays, Kronecker
using OrdinaryDiffEq, DiffEqBase, DiffEqCallbacks

include("types.jl")
export AbstractGate, HaarGate, LazyHaarGate, hamiltonian, RowGate, LazyRowGate, generate_random_circuit, generate_lazy_random_circuit

include("timeevolution.jl")
export compute_tstops, time_evolution

end
