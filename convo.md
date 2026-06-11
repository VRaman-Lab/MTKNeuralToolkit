I want to use modelingtoolkit.jl to build a julia package that builds and connects neuron components to build networks of biophysical neurons.

MTK allows me to treat things like an electrical circuit: here are some excerpts from what I've done.
I want to plan the package before getting into more detail. First thing to think about is that I sometimes want the soma to track calcium concentrations. Some ion channels might be calcium sensitive, or push calcium into the soma. Here are examples:


Problem is calcium is not fully tracked in such models: it comes from a limitless external reservoir. SO I want ideas on how I could plan out the components so that I could easily add calcium ( and maybe even other ions if it's generalisable) while keeping an easy to use codebase
