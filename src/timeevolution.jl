import Base: resize!
resize!(M::ElasticMatrix, N::Integer) = resize!(M, size(M,1), div(N,size(M,1)))

"""
    hermitianize!(x)

Extract the Hermitian part of `x`.
# Arguments
* `x`: some matrix.
"""
function hermitianize!(x::AbstractMatrix)
    ishermitian(x) && return x
    n = LinearAlgebra.checksquare(x);
    @inbounds @views for j = 1:n, i = 1:j
        x[i,j] = (x[i,j] + conj(x[j,i])) / 2.
        x[j,i] = conj(x[i,j]);
    end
    return x
end

"""
    compute_tstops(circuit, t0=0.0)

Compute all callback and gate switching times of the circuit.
"""
function compute_tstops(circuit::AbstractArray{G}, t0::Float64 = 0.0) where G<:AbstractGate
    # Times at which a PCA step is performed
    tstops  = Float64[t0]
    # Times at which the current gate changes (change of Hamiltonian)
    tswitch = Float64[]

    t = t0
    for gate in circuit
        push!(tstops, [t:gate.dt:t+gate.T;][2:end]...)
        push!(tswitch, t + gate.T)
        t += gate.T
    end

    return tstops[1:end], tswitch[1:end-1]
end

"""
    time_evolution(L0, circuit, J; algo=Tsit5(), ϵ=1e-5, tol=1e-9, fout=missing)

Compute the full time evolution of an initial state `L0` through a `circuit` under
the action of dissipators `J`.
"""
function time_evolution(L0::ElasticMatrix, circuit::AbstractArray, J::AbstractArray; algo=Tsit5(), ϵ=1e-5, tol=1e-9, fout=missing)
    N = size(L0,1)          # Size of the full Hilbert space
    M = size(L0,2)          # Size of the corner
    tstops, tswitch = compute_tstops(circuit)
    i_g = 1                 # Current gate index
    h = circuit[i_g].dt     # Current timestep

    # "kraus" operators (K_i = sqrt(h) * k_i)
    ks = map(j -> j.data, J)

    # Compute the non-Hermitian term of the Hamiltonian
    nh_term = zero(first(ks))
    for k in ks
        nh_term -= 0.5 .* k'k
    end
    # Initialise the non-Hermitian Hamiltonian
    δk0 = -im.*hamiltonian(circuit[i_g])::typeof(nh_term) + nh_term # Generator of the pseudounitary dynamics

    # results from fout
    res_fout = ismissing(fout) ? nothing : [fout(0.0, L0)]

    # Caches
    KL = ElasticArray{ComplexF64}(undef, N, (length(ks)+1) * M)
    KL.= zero(eltype(KL))
    # Precompute the dissipative Kraus trajectories at t+h
    for i in 1:length(ks), j in 1:M
        KL[:,i*M + j] .= sqrt(h) .* ks[i]*L0[:,j]
    end

    # Pseudounitary time evolution
    function f!(dL, L, p, t)
        mul!(dL, δk0, L)
    end

    # Function called at each callback, performs PCA over the Kraus trajectories
    # generated during the previous time step
    function affect!(integrator)
        if integrator.t in tswitch
            # Switch to next quantum gate
            i_g += 1
            # Adapt timestep
            h = circuit[i_g].dt
            # Adapt the Hamiltonian to match the new gate
            δk0 = -im.*hamiltonian(circuit[i_g])::typeof(δk0) + nh_term
        end
        # Add missing pseudounitary trajectories
        @inbounds @views KL[:,1:size(integrator.u,2)] .= integrator.u
        # Reinitialise the integrator state
        integrator.u .= zero(ComplexF64)

        # Build the low-rank representation of the density matrix
        ρ = KL'KL
        hermitianize!(ρ)
        _λs, cs = eigen(ρ)
        λs = real.(_λs)
        λs[λs .< 0.0] .= 0.0
        # Regularise
        λs ./= sum(λs)

        perm = sortperm(λs; rev=true)
        # New corner dimension
        M = findfirst(trace->1.0-trace < ϵ, accumulate(+,λs[perm]))
        # map to kept columns
        σ = perm[1:M]

        # Resize the integrator state vector and caches to match the new corner dimension
        resize!(integrator, N*M)

        # Initialise the integrator state for the timestep to come
        for (i, σi) in enumerate(σ)
            # Non-allocating equivalent to :
            # integrator.u[:,i] .= sqrt(λs[σi]) .* normalize(droptol!(sparse(KL * cs[:,σi]),tol))
            v = @view integrator.u[:,i]
            mul!(v, KL, view(cs, :, σi), one(eltype(v)), zero(eltype(v)))
            v[abs.(v) .< tol] .= zero(eltype(v))
            normalize!(v)
            v .*= sqrt(λs[σi])
        end

        # Apply fout to the current integrator state
        !ismissing(fout) && push!(res_fout, fout(integrator.t, copy(integrator.u)))

        # Precompute the dissipative Kraus trajectories at t+h
        resize!(KL, N, (length(ks)+1) * M)
        for i in 1:length(ks), j in 1:M
            KL[:,i*M + j] .= sqrt(h) .* ks[i]*integrator.u[:,j]
        end

        integrator
    end

    # Callback that deals with the periodic call of the PCA procedure
    cb = PresetTimeCallback(tstops[2:end],affect!)

    # Build the ODE problem
    pb = ODEProblem(f!, L0, extrema(tstops))

    L = solve(pb, algo, callback=cb, save_on=false, save_end=true).u[2]

    if ismissing(fout)
        return L
    else
        L, res_fout
    end
end

time_evolution(L0::StateVector,    circuit::AbstractArray, J::AbstractArray; kwargs...) = time_evolution(L0.data, circuit, J; kwargs...)
time_evolution(L0::AbstractVector, circuit::AbstractArray, J::AbstractArray; kwargs...) = time_evolution(reshape(L0,length(L0),1), circuit, J; kwargs...)
time_evolution(L0::AbstractMatrix, circuit::AbstractArray, J::AbstractArray; kwargs...) = time_evolution(ElasticMatrix(L0), circuit, J; kwargs...)
