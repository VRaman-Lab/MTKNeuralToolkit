ANN: MTKNeuralToolkit.jl- acausal modelling of biophysical neurons and neural circuits

Note: I'm happy with the core functionality but there is much to add and I'm keen for any interested parties to help out! This is a side project where I'm interested in using the simulator for my own purposes, but not in independently turning this into a new version of NEURON.

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

- Use adaptive timestep ODE solvers, and differentiate through them. Much better for simulation speed I hypothesise.
- Take advantage of sparse jacobians for simulation and AD for free, by just going `ODEProblem(mtk_sys, ...; jac=true, sparse=true)`. Jaxley have made their own [tridiax](https://github.com/jaxleyverse/tridiax) for tridiagonal systems but increasing coverage would be a huge maintenance cost
- Differentiate through a more flexible set of models, such as those with synaptic dynamics. Or whatever you want really, as long as `@mtkcompile` produces a ModelingToolkit system
- Not have to maintain
 
- Differentiable using adaptive timestep simulations (unlike Jaxley). You can use all the DifferentialEquations.jl tricks like automatically finding and exploiting jacobians and sparsity for faster simulation and autodiff.

And then pragmatically, building on MTK gives a much smaller codebase to maintain.

(like Jaxley with the Diffrax/Jax ecosystem)










### Compared to NEURON
- **Doesn't maintain an entire ecosystem of ODE solvers and autodiff engines.** All of that is passed to the rest of the SciML ecosystem.
- **Acausal.** The codebase is tiny and highly configurable. You just make your own ion channel / synapse / etc as a ModelingToolkit `@component`. The package is simply for hooking ion channels to compartments, and compartments to networks via Kirchhoff's laws.
- **Differentiable.** Gradient descent, parameter estimation, and optimization are possible out of the box—not because this package implements them, but because it inherits the general SciML stack. 
- **Presumably much faster runtime.** Since we neuroscientists are not as good at numerical analysis as numerical analysts, leaning on `OrdinaryDiffEq.jl` gives us access to state-of-the-art solvers for free.

### Compared to Jaxley
You could say all these advantages (differentiability, not maintaining a numerical stack) exist in the recent package [Jaxley](https://github.com/jaxleyverse/jaxley), which is a differentiable neural simulator written in Python/JAX. Jaxley is much more mature than this package. However, there are tradeoffs between using the Diffrax/JAX ecosystem and the SciML ecosystem. For biophysical neural circuits, I feel the long-term advantages are in favor of SciML. For instance, with this package you can:

- **Use adaptive timestep ODE solvers, and differentiate through them.** Jaxley currently relies on fixed-timestep solvers (`fwd_euler`, `bwd_euler`, `crank_nicolson`, `exp_euler`). With MTK, you can easily use high-order, adaptive, implicit stiff solvers (like `Rosenbrock23` or `Rodas4`), which is much better for simulation speed and stability on biophysical ODEs. You can differentiate through these adaptive solves using the mature suite of adjoint methods in `SciMLSensitivity.jl`.
- **Take advantage of sparse Jacobians for simulation and AD for free.** You just go `ODEProblem(mtk_sys, ...; jac=true, sparse=true)`. Jaxley has made their own [tridiax](https://github.com/jaxleyverse/tridiax) package for tridiagonal systems, which is very fast for standard compartmental cables. But if you want to introduce non-local couplings, weird gap junctions, or complex global chemical synapse dynamics, those tridiagonal assumptions break down. Increasing coverage would require a huge maintenance cost on their end. MTK's automatic sparsity detection handles arbitrary structures out of the box.
- **Differentiate through a much more flexible set of models.** You can implement arbitrary synaptic dynamics, calcium tracking with Nernst potentials, or continuous STDP rules. You can do whatever you want really, as long as `@mtkcompile` produces a ModelingToolkit system.

## What's in the box right now?

The package currently supports:
- Scalar and Vectorized topologies (so you can simulate a single neuron, or a population of 1,000 neurons using array broadcasting to save compile time).
- Pre-built channels like `GenericChannel`, `CaVChannel`, and `KCaChannel`.
- A library of standard models, including Hodgkin-Huxley, Morris-Lecar, FitzHugh-Nagumo, Liu Calcium Neuron, and the Prinz Stomatogastric Ganglion (STG) network.
- A variety of synapse types: Exponential, Alpha, NMDA, Cholinergic, Glutamatergic, and continuous STDP.
- Acausal gap junctions for multi-compartment cable models.

If you are interested in biophysical modeling, acausal systems, or just want to hack on some neuroscience models in Julia, I'd love for you to check it out, try the examples, and open issues/PRs!

**Repo:** [Link to your GitHub Repo]
**Docs/Examples:** [Link to Docs]
