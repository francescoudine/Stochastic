abstract type OdeSolver end

struct Euler <: OdeSolver end
struct EulerCromer <: OdeSolver end
struct Midpoint <: OdeSolver end
struct Leapfrog <: OdeSolver end
struct Verlet <: OdeSolver end

struct OdeSolution{R,T}
    t::R
    x::Vector{T}
    v::Vector{T}
end

OdeSolution(T, r) = OdeSolution(r, Vector{T}(undef, length(r)), Vector{T}(undef, length(r)))

init(::Type{T},ϕ,x0,v0,Δt) where T<:OdeSolver = x0, v0
init(::Type{Leapfrog},ϕ,x0,v0,Δt) = x0, v0+ϕ(x0)*Δt/2

function solve(solver::Type{T}, ϕ, x0::F, v0::F, tspan::Tuple{eltype(F),eltype(F)};
               Δt=0.1, verbose::Bool=true) where {T <: OdeSolver, F}
    xn, vn = init(solver, ϕ, x0, v0, Δt)
    tinterval = tspan[1]:Δt:tspan[2]
    sol = OdeSolution(typeof(x0), tinterval)
    for n in eachindex(tinterval)
        sol.x[n], sol.v[n] = xn, vn
        xn, vn = update(solver, ϕ, xn, vn, Δt)
        verbose && println("n=$n tₙ = $(tinterval[n]) xₙ = $xn vₙ = $vn")
    end
    return sol
end

function update(::Type{Euler},ϕ,x,v,Δt)
    v1 = v + ϕ(x)*Δt
    x1 = x + v * Δt
    return x1, v1
end

function update(::Type{EulerCromer},ϕ,x,v,Δt)
    v1 = v + ϕ(x)*Δt
    x1 = x + v1*Δt
    return x1, v1
end

function update(::Type{Midpoint},ϕ,x,v,Δt)
    v1 = v + ϕ(x)*Δt
    x1 = x + (v + v1)*Δt/2
    return x1, v1
end

function update(::Type{Leapfrog},ϕ,x,v,Δt)
    x1 = x + v * Δt
    v1 = v + ϕ(x1)*Δt
    return x1, v1
end

function update(::Type{Verlet},ϕ,x,v,Δt)
    v += ϕ(x)*Δt/2
    x += v*Δt
    v += ϕ(x)*Δt/2
    return x, v
end

function energy(sol, ω)
    t,x,v = sol.t,sol.x,sol.v
    E0 = 0.5*((ω*x[1])^2 + v[1]^2)
    ((ω*x).^2 + v.^2 .- E0)/2E0
end
