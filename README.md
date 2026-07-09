# MTKNeuralToolkit

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://vraman-lab.github.io/MTKNeuralToolkit.jl/dev/)
[![Build Status](https://github.com/Dhruva2/MTKNeuralToolkit.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Dhruva2/MTKNeuralToolkit.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Julia](https://img.shields.io/badge/julia-1.10%2B-blue)](https://julialang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


# Building Your Own Neurons & Components in MTKNeuralToolkit

MTKNeuralToolkit is a framework for wiring up your own custom biophysical components using acausal modeling. The pre-built stuff like `GenericChannel` and `GateSpec` is there to save you time, but everything is just an MTK `System`. If a pre-built component doesn't fit your e.g. custom strange ion channel, you are fully empowered to build your own from scratch.

The [examples repository](https://vraman-lab.github.io/MTKNeuralToolkit.jl/dev/) has an ordered set of tutorials. Go through them to learn by example.

*Note 1: I'm happy with the core functionality but there is much to add and I'm keen for any interested parties to help out! This is a side project where I'm interested in using the simulator for my own purposes, but not in independently turning this into a new version of NEURON. *

*Note 2: If you see any architectural issues with the core functionality please let me know! I'd like to build a really solid base for students and others to then add lots of sugar to. *

## The context

You could divide the space of neuron simulators into
1. clock-driven packages designed for huge groups of integrate and fire neurons (eg Brian2, NEST). They have a fixed timestep, and are optimised to update huge numbers of voltage reset events each timestep
2. biophysical simulators designed to simulate entire voltage spikes along geometries, using differential equations to model ion channels, compartments, and synapses (e.g. NEURON, Jaxley).

`MTKNeuralToolkit` belongs to the second category. But it's built on ModelingToolkit which gives it important differences. If we compare first to NEURON:

- Doesn't maintain an entire ecosystem of ODE solvers and autodiff engines, since that's all passed to the rest of the SciML ecosystem .
- Acausal, so the codebase is tiny and configurable: you just make your own ion channel / synapse / etc as a ModelingToolkit `@component`. The package is just for hooking ion channels to compartments, and compartments to networks.
- Differentiable, so gradient descent etc is possible (not from this package, just the general SciML stack)
- Presumably much faster, since we neuroscientists are not as good at numerical analysis as numerical analysts?

You could say all these advantages exist in the recent package [Jaxley](https://github.com/jaxleyverse/jaxley), which is another differentiable neural simulator written in Python/Jax/Diffrax and is much more mature than this package. However there are tradeoffs between using Diffrax/Jax and SciML as your AD-friendly ODE solving stack. For biophysical neural circuits, I feel the long-term advantages are in favour of SciML. For instance in this package you can:

- Use adaptive timestep ODE solvers, and differentiate through them. Much better for simulation speed I hypothesise. Jaxley is restricted to (`fwd_euler`, `bwd_euler`, `crank_nicolson`, `exp_euler`), presumably to maintain differentiability.
- Take advantage of sparse jacobians for simulation and AD for free, by just going `ODEProblem(mtk_sys, ...; jac=true, sparse=true)`. Jaxley have made their own [tridiax](https://github.com/jaxleyverse/tridiax) for tridiagonal systems, but this presumably won't include e.g. non-local couplings. Increasing coverage would be a massive maintenance cost for them, whereas we get generic MTK sparsity detection for free.
- Differentiate through a more flexible set of models, such as those with synaptic dynamics, calcium tracking with nernst potentials, or continuous stdp rules. Or whatever you want really, as long as `@mtkcompile` produces a ModelingToolkit system
- Not have to maintain
 
- Differentiable using adaptive timestep simulations (unlike Jaxley). You can use all the DifferentialEquations.jl tricks like automatically finding and exploiting jacobians and sparsity for faster simulation and autodiff.

And then pragmatically, building on MTK gives a much much smaller codebase to maintain, and it's much much easier to build functionality. I'm far from a top level coder, but I programmed this version from scratch (taking full advantage of useful and well programmed existing attempts by my excellent students Elouan Simmoneau and Ella Bennison, who had little previous knowledge of biophysics, neuroscience or circuit theory) in about a week and a bit of solid work. With help and occasional hindrance from GLM5.2. That would be completely impossible in another framework I think. 

### The context (Julia ecosystem)

*please let me know if i missed anything out!*

- [SpikingNeuralNetworks.jl](https://juliasnn.github.io/SpikingNeuralNetworks.jl/dev/) looks very cool, much more mature, and I can't comment much on it as I just found it. Big difference is it's not using the SciML stack for simulation it seems?


- I had a go at doing this previously with students Andrea Hincapie and Pavel Piekarz doing some heroic coding: [neuronbuilder.jl](https://discourse.julialang.org/t/ann-neuronbuilder-jl-a-differentiable-neuronal-simulator/78743). I stopped as it was built on an earlier version of MTK and we had to make our own hacky implementations for connecting and extending components, so got more complicated than it was worth. The current codebase is much cleaner and feels more easily extendable.

- Wiktor Phillips and the Julia lab were building [conductor](https://github.com/wsphillips/Conductor.jl/tree/main), but feels like work stopped...?

- 

### Lessons learnt / cons

*(might be relevant if you use or develop MTK. Or maybe you could point me to functionality that already does this stuff in MTK)*

- `@mtkcompile` doesn't scale for large systems. So I had to use a ...hack? Made my own vectorised versions of Pin, OnePort, TwoPort etc in the MTK standard library. Then you can make a vector of components (eg Hodgkin huxley neurons) with little compile overhead. **Unfortunately** you need to define your components twice to make use of this: scalar and vectorised case (i guess you could do vectorised with N=1 too). I tried to make a macro that builds a vectorised component out of a scalar one, and even got it to work a bit, but ditched it as it was getting ugly and felt fragile. **But I'd love it if MTK developers made something like this for me!!**. Right now you have to define components twice: scalar and vector. Which is ugly.

- I wanted to be very correct originally, and eg build electric ion channels as a battery (reversal) in series with a nonlinear resistor (ion channel gates). It just feels elegant and correct. But then each neuron has too many pins and connecting equations and `@mtkcompile` isn't happy. **It would be great to have a functionality that 'simplifies out' the pins and observed variables from components** so I could build properly and then have a final result with low algebraic burden.

# Rough outline of how it works

---

## 1. `OnePort`, `TwoPort` and `extend`

Most electrical components, e.g. capacitor, ion channels, are built on the idea of an `OnePort`. An `OnePort` has a positive pin (`p`) and a negative pin (`n`), a voltage across them (`v`), and a current flowing through it (`i`).

If you are building something that connects two different compartments—like a GapJunction or a ChemicalSynapse—you will use a TwoPort. A TwoPort gives you two sets of pins (p1, n1, v1, i1 and p2, n2, v2, i2).

When you build a custom channel, you don't have to worry about connecting pins manually. You just define the math for your channel, put it in an `OnePort`, and use `extend`. The `build_compartment` function will handle connecting it to the rest of the cell using Kirchhoff's laws automatically.

Here is the skeleton for making your own custom channel:

```julia
using ModelingToolkit: t_nounits as t, D_nounits as D, @named, @variables, @parameters, @component, System, Equation, SymbolicT, extend
using ModelingToolkitStandardLibrary.Electrical: OnePort

@component function MyWeirdChannel(; name, g=1.0, E_rev=10.0)
    # 1. Grab your standard OnePort
    @named oneport = OnePort()
    @unpack v, i = oneport
    
    # 2. Define your parameters and state variables
    @parameters g=g E_rev=E_rev
    @variables x(t)=0.0
    
    # 3. Write your dynamics. 
    # IMPORTANT: The current equation MUST be assigned to `i`.
    eqs = Equation[
        D(x) ~ (1.0 - x) / 10.0,  # some custom gating variable
        i ~ g * x * (v - E_rev)   # the toolkit handles the KCL signs for you!
    ]
    
    # 4. Extend the OnePort and return it
    return extend(System(eqs, t, [x], [g, E_rev]; systems=System[], name=name), oneport)
end
```

Because you hooked it to an `OnePort`, you can just pass this directly to `build_compartment` and it will snap right in alongside `GenericChannel`s.

---

## 2. `GateSpec` and `GenericChannel` are just conveniences

You don't *have* to use `GateSpec`. It's just a struct we provide to save you from writing the same `alpha * (1 - x) - beta * x` boilerplate over and over. 

If you want to use it, just remember it expects an `(alpha, beta)` tuple:

```julia
struct GateSpec
    name::Symbol
    power::Integer
    ic::AbstractFloat
    dynamics::Function # Must return (alpha, beta)
end
```

Most textbooks use `(inf, tau)` instead of `(alpha, beta)`. Don't manually convert them! Just use a helper function:

```julia
# Convert (inf, tau) -> (alpha, beta)
InfTau(inf_fn, tau_fn) = v -> (inf_fn(v) ./ tau_fn(v), (1.0 .- inf_fn(v)) ./ tau_fn(v))

# For calcium-dependent gates that need both v and ca:
InfTauCa(inf_fn, tau_fn) = (v, ca) -> (inf_fn(v, ca) ./ tau_fn(v), (1.0 .- inf_fn(v, ca)) ./ tau_fn(v))
```

If your channel has dynamics that don't fit the standard `g * m^power * (v - E_rev)` shape (like the FitzHugh-Nagumo model), skip `GenericChannel` entirely and use the `OnePort` trick above. You are free to do whatever you want inside your `OnePort`'s equations.

---

## 3. The Calcium Sign Trap

If you are building custom calcium channels or calcium pools, you need to know about MTK's `Flow` connector convention.

When you `connect()` two flow ports, MTK enforces that their flows sum to zero (KCL for chemicals). The `CalciumPool` tracks concentration using:
`D(Ca) = decay_term + pool.J_Ca`

If a calcium channel pushes calcium *out* of its port (positive `J_Ca`), the pool receives a *negative* `J_Ca` because they sum to zero. If `pool.J_Ca` is negative, `Ca` drops. If `Ca` drops to zero, your Nernst equation `log(Ca_out / Ca)` goes to infinity, and your neuron voltage will instantly shoot to ±infinity.

**The Fix:** 
When translating inward calcium currents (where the electrical current `i = g*(v - E_rev)` is negative), your `conversion_factor` in `CaVChannel` must be **positive**.

A positive `conversion_factor` multiplied by a negative `i` gives a negative channel `J_Ca`. The `connect()` law flips it to a positive `pool.J_Ca`, safely increasing your Calcium concentration and creating the stable negative feedback loop you want.


```julia
# WRONG: conversion_factor = -0.94
# This causes calcium to drain to 0, blowing up the simulation.

# CORRECT: conversion_factor = 0.94 / 20.0  (assuming tau_Ca = 20.0)
@named cav = CaVChannel(g=3.0, conversion_factor=0.047, gates=my_gates, Ca_out=3000.0, nernst_factor=12.19)
```

---

## 4. Don't fight the circuit signs

When porting biological equations, you'll often see:
`I_Na = g * m^3 * h * (E_Na - V)`

You might be tempted to write `i ~ g * (E_rev - v)` in your custom `OnePort` to match the textbook. **Don't.**

Always write it as `i ~ g * (v - E_rev)`. 

Why? Because `build_compartment` connects your channel to the membrane capacitor. By KCL, the current flowing *out* of the capacitor equals the current flowing *in* to the channel. The toolkit's capacitor equation is `D(v) = i_cap / C`. The maths automatically resolves to `D(v) = - sum(i_channels) / C = sum(g * (E_rev - v)) / C`. It perfectly matches your biology without you having to manually distribute negative signs. Stick to standard electrical convention inside your components and let the acausal framework do the algebra!

Here are a couple of new sections you can drop right into the guide. They cover vectorized components and the two ways to wire up synapses, using the scripts you provided as a reference.

---

## 5. Vectorized Components (Saving Compilation Time)

If you want to simulate a network of 1,000 identical neurons, you *could* build 1,000 scalar compartments and let MTK compile them. But MTK compilation scales with the number of equations, so building massive scalar networks will make your `@time mtkcompile(net.sys)` take forever and eat all your RAM. 

To fix this, the toolkit supports **Vectorized** components. Instead of passing `topology=Scalar()`, you pass `topology=Vectorized(N)`. The toolkit will automatically create array variables (e.g. `v[1:N]`) and use broadcasted math (`./`, `.*`), collapsing 1,000 equations into a single vectorized equation that MTK compiles instantly. 

You build a vectorized compartment exactly the same way you build a scalar one—just swap the topology!

```julia
N = 100
top = Vectorized(N)

@named soma = Capacitor(topology=top, C=1.0)
@named na_ch = GenericChannel(topology=top, g=120.0, E_rev=50.0, gates=sodium_gates)
@named k_ch  = GenericChannel(topology=top, g=36.0, E_rev=-77.0, gates=potassium_gates)

# This single compartment represents 100 identical cells bundled together
pop = build_compartment(soma, [na_ch, k_ch]; name=:pop, V_init=-65.0, topology=top)
```

*Tip: Always make sure your custom gate functions use element-wise operators (`./`, `.*`, `.^`) so they work smoothly when passed a vectorised voltage `v`.*

---

## 6. Wiring Synapses: Blocks vs. Point-to-Point

Once you start using vectorised compartments, you need a way to wire them together. The toolkit gives you two ways to do this depending on whether you want dense population connectivity or specific cell-to-cell wiring.

### Option A: Synapse Blocks (Dense / All-to-All)
If you have a weight matrix `W` (size `N_post x N_pre`) defining the connectivity between two populations, use `build_synapse_block`. It creates a single `VectorizedExpSynapse` that computes `I_syn = W * s` via matrix multiplication. This is extremely fast.

```julia
W_EI = 0.1 .* rand(N_I, N_E)   # Weight matrix from E pop to I pop

# Automatically wires pop_E's voltages to pop_I's synaptic currents
syn_EI = build_synapse_block(pop_E, pop_I, W_EI; name=:syn_EI, E_rev=0.0)

synapse_specs = [syn_EI]
net = build_acausal_network([pop_E, pop_I]; synapse_specs=synapse_specs)
```

### Option B: Point-to-Point `SynapseSpec`
Sometimes you don't want a dense matrix. Maybe you just want cell 1 to connect to cell 2 inside a single vectorized population. You can manually pick out the specific array indices from the `interfaces` and pass them to a scalar `ExpSynapse` via a `SynapseSpec`.

```julia
@named syn_1to2 = ExpSynapse(g_max=2.0, τ=5.0, E_rev=0.0)

# Explicitly grab index 1 for pre_V, index 2 for post_V, and index 2 for the post_I_syn
synapse_specs = [
    SynapseSpec(hh.interfaces.V[1], hh.interfaces.V[2], hh.interfaces.I_syn[2], syn_1to2)
]

net = build_acausal_network([hh]; synapse_specs=synapse_specs)
```

The `build_acausal_network` function is smart enough to look at your `SynapseSpec`s. If multiple synapses converge on the exact same target (e.g., `1→2` and `4→2`), it will automatically sum their currents into a single equation for `I_syn[2]` before grounding the unused indices for you.
