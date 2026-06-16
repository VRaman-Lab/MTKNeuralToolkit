# Planning

- Existing code is fragile (e.g. build channel logic). Need it cleaner.
- Adding ion tracking (calcium) has made everything complicated and fragile. Not clear we are doing it in the best way
- Don't want to be using callbacks for autodiff ideally. Don't have to for LIF

- I think it's best to have a fresh start. And to add calcium functionality once we have a good system working and differentiating for voltages only. 
- I've built some scripts that show how i'd like build channel and lif to be implemented in the package. Consider and critique them.





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



## 








