Project Context: MTKNeuralToolkit
MTKNeuralToolkit is a Julia package for building biophysical Spiking Neural Networks (SNNs) using ModelingToolkit.jl (MTK) and Symbolics.jl. The target audience is computational neuroscientists who need biological accuracy, heterogeneous microcircuits, and SciML integration (e.g., parameter optimization via ForwardDiff), rather than pure brain-scale throughput.

Architectural Philosophy
The toolkit follows a Hybrid Architecture to balance biological modularity with computational performance:

Biological Modularity: Neurons and ion channels are defined using intuitive, scalar, acausal components (OnePort, Pin, connect()).
Network Performance: Networks are scaled using native array broadcasting and sparse matrix math, bypassing the poor scaling of MTK's acausal connect() graphs for large $N$.
Short & Medium-Term Implementation Goals
Goal 1: Implement vectorize_system (The Auto-Vectorizer)
Write a function that takes a scalar MTK System (e.g., a multi-compartment cell) and an integer $N$, and returns a natively vectorized System where all variables are arrays of size $N$ and all equations use broadcasting (.*, ./).

Technique: Substitute scalar variables with array variables, convert the expression to a Julia Expr, use MacroTools.postwalk and @capture to inject broadcasting dots for math operations (blacklisting structural ops like Differential), and use Symbolics.parse_expr_to_symbolic to safely rebuild the symbolic equations without eval.
Location: Create a new file src/vectorization.jl and export it.
Goal 2: Refactor Synapse Architecture to Matrix Blocks
Transition away from scalar/cloned synapses. Synapses must be defined at the matrix level because they scale pairwise.

Technique: Create @component functions (e.g., AlphaSynapseMatrix) that accept sparse weight matrices W::SparseMatrixCSC and define states as 2D arrays S[1:N_pre, 1:N_post]. Use sparse matrix multiplication to compute post-synaptic currents I_out.
Stateful Synapses: Support complex internal states (e.g., local calcium, continuous STDP weights) by making W a state variable with initial conditions rather than a parameter.
Goal 3: Build the CellPopulation and Network Tiers
Update the hierarchy in src/connections.jl:

Compartment (Scalar/Acausal): Keep existing build_compartment.
Cell (Scalar/Acausal): Keep existing build_cell.
Cell Population (Vectorized): A new tier that takes a scalar Cell and $N$, utilizing the vectorize_system function.
Network (Matrix-Based): A new build_network function that takes a list of CellPopulations and a list of SynapseMatrix blocks. It wires them by summing the matrix current outputs (I_out) into the populations' external current arrays (I_ext).
Deprecation: Phase out the clone_compiled_cell engine in connections.jl in favor of this new population/matrix pipeline.
Goal 4: Natively Vectorized Spiking/Event Handling
For networks that require discrete spikes (LIF, STDP), do not rely on auto-vectorizing scalar events. Instead, provide natively vectorized neuron templates.

Technique: Define @variables V(t)[1:N] directly. Use ModelingToolkit.ImperativeAffect with continuous_events to handle vector events. Inside the affect function, use array broadcasting (e.g., ifelse.(x.V .>= v_th, V_reset, x.V)) to reset only the spiked neurons without needing $O(N)$ scalar event loops.
Essential Documentation for LLM Context
If an LLM is working on this codebase, it should review the following documentation files (which were provided during the architectural design process):

src/manual/arrays.md (Symbolics.jl):
Explains the difference between arrays of symbolics and SymbolicArray (O(1) representation). Crucial for understanding why natively vectorized systems scale better.
src/manual/parsing.md (Symbolics.jl):
Documents parse_expr_to_symbolic and @parse_expr_to_symbolic. Required for Goal 1 to safely convert manipulated Julia Expr trees back into MTK Equations without causing world-age issues.
src/pattern-matching.md (MacroTools.jl):
Documents postwalk and @capture(f_(xs__)). Required for Goal 1 to elegantly match function calls in the AST and inject broadcasting dots (.+, .*).
src/manual/events.md (ModelingToolkit.jl):
Documents continuous_events and ImperativeAffect. Required for Goal 4 to understand how to pass array states into an affect function and use Setfield.@set! to conditionally reset spiking neurons.
src/tutorials/auto_parallel.md (Symbolics.jl):
Demonstrates how tracing array functions and sparse Jacobians works. Helps understand the performance benefits of the matrix-block synapse architecture.



## issues


**Title:** `Pre()` operator on Symbolic Array indices crashes `mtkcompile` with `matching non-exhaustive` / `setsym` errors

**Description:**
When defining a `continuous_event` affect that updates an element of a Symbolic Array (`@variables S(t)[1:N, 1:N]`) using the `Pre()`
operator, `mtkcompile` fails during structural simplification.

MTK appears unable to namespace or re-initialize array indexing expressions (e.g., `S[j, i]`) when they are wrapped in the `Pre`
operator. This currently blocks the creation of scalable, vectorized event-driven components (like neural populations) and forces the
use of `ImperativeAffect` or programmatic scalar variable generation as workarounds.

**Minimal Reproducible Example:**
```julia
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq

function VectorizedSynapseBug(; name, N::Int)
    @variables v(t) I(t) S(t)[1:N, 1:N]
    @parameters W[1:N, 1:N] = ones(N, N)

    # Add an algebraic variable I to trigger setsym during init
    eqs = [D(v) ~ -v, D(S) ~ -S, I ~ sum(S)]

    # Event updating an array index using Pre()
    affect = [S[1, 1] ~ Pre(S[1, 1]) + W[1, 1]]
    events = [[v ~ 0.0] => affect]

    return System(eqs, t; continuous_events=events, name=name)
end

@named sys = VectorizedSynapseBug(N=2)
sys_compiled = mtkcompile(sys)

prob = ODEProblem(sys_compiled, [], (0.0, 1.0))
 ...

LoadError: Invalid symbol Pre((S(t))[2, 1]) for `setsym`


**Context & Workaround:**
I am developing a neural modeling package. To achieve $O(N)$ scaling for all-to-all synaptic connections, we generate $N$ events (one
per pre-synaptic neuron) that each update a row of a weight/state matrix `S`

## Proposed core logic:

Need to make sure calcium sensitive neurons and channels are acknowledged. How?

### Option 1: hardcode an IonicPort that generalises across ions. 


Soma has a calcium potential. Calcium is a flow variable
Specifically, build an IonicOnePort() something like

@connector function IonicPort(; name)
    vars = @variables begin
        C(t), [description = "Concentration (e.g., mM)"]
        i(t), [description = "Ionic current component (e.g., mA or pA)"]
    end
    # MTK's connector system automatically sums 'i' at a junction 
    # and ensures 'C' is equal across connected ports.
    ModelingToolkit.System(vars, t, name=name; flows=[i])
end




### Extend(oneport)
Soma
Voltage-sensitive ion channel


### Extend(twoport)
Synapse



## 








