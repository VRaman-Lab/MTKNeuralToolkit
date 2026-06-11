# Notes

Optimisation stuff like Lux is in the package. It should be in the scripts env


Don't need callbacks for LIF. need ifelse:
https://docs.sciml.ai/ModelingToolkitStandardLibrary/stable/tutorials/custom_component/


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



### Connections






### Improving to new MTK version

https://juliahub.com/blog/what-s-new-with-modelingtoolkit

can put an input as as parameter (see above)


### questions

Why do we have nonlinearsolve in plot_LIF?
Where is this explicit channel being used? Doesn't seem used in HH or Liu

### Code cleaning for Ella/Elouan

Write a short summary of the different functions. eg i'm looking at src/Electrical/utils.jl :

There are several build_channel and build_channel_explicit methods. In what contexts are they used? What types of channel need which?
Something that would help is to add some 'templates'. What do we require a channel to have, programmatically? A channel.p and a channel.n? 
