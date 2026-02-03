
utils.md

1.There are three versions of build_channel. There should be one.
2. The CaS gates etc are not gates. They are entire channels. Then they are put in series with a 0 reversal in scripts/plot_Liu.jl. This is confusing and wrong. If we make an entire channel with a single @mtkmodel, then we should call it a channel not a gate.

3. You have two methods for build_neuron that only differ in their choice of input. Lots of copy pasted code is a code smell. Better pattern is:

build_neuron(soma, channels) = build_neuron(soma  input = RealInput(u ~ 0); channels)

function build_neuron(soma, input::TypeOfNonTrivialInput; channels)
...same as before
end


This will avoid lots of downstream complications. For instance in build_prinz 

   if input === nothing
        neur = build_neuron(fn;  channels = [KCa, Na, CaS, CaT, K, DRK, H, Leak])
    else
        neur = build_neuron(fn, input;  channels = [KCa, Na, CaS, CaT, K, DRK, H, Leak])
    end
    return(neur)


If you do the multiple dispatch properly these and other if statements are unnecessary.



4. add_synapse_no_odesystem doesn't appear to be used. But same as before. If you have a function with a modification ("_no_odesystem") it's a code smell. You probably need two methods. And one method should call the other method to avoid buggy code duplication


So the following avoids duplication:


function add_synapse(relevant arguments)

code for add_synapse_no_odesystem

end

function add_synapse(relevant arguments)

call previous add_synapse
do the odesystem stuff

end

If "relevant arguments" are problematically the same for both of these methods, then the first method should have a different name.



script_utils.md

Why does this file exist? build_network is core functionality and should be in the src folder. If you want to separate it from the other utils then you could call it something else, e.g. build_network_utils. 


plot_hh_network.md

1. None of the three workflows for building a network are ideal. We want neurons = [
  build_neuron(...)
  build_neuron(.......)
]

Then we can access neurons by their order in the array. I promise this will make life easier for postprocessing stuff.

A dictionary means there is no ordering which will make life harder when doing advanced postprocessing. It means you have to define the names twice (name => build_neuron(;name = name)) which is confusing. Accessing neurons by an ambiguous name (which one of the two? What if we had to add a number suffix at the end to the name?) is bug prone.


To get neuron voltages you are currently using a regexp, which is extremely fragile and against the spirit of getting ModelingToolkit funcitonality to do complicated accessing of variables for you. But let's save fixing these things to when the structure of networks is finalised.

https://docs.sciml.ai/ModelingToolkit/dev/API/variables/#Miscellaneous-metadata

2. To get the voltage of the AB neuron, I take:
network.s_ABLP.AB.AB.v

Why are there two layers of ABs? This is a code smell and not ideal. Can we add the neurons as direct subsystems of the network?


There is more cleaning to do but let's get the above done first :)
