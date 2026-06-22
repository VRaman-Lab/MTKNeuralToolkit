
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








