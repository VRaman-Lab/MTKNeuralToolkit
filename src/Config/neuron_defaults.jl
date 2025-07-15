@kwdef struct PrinzConfig

   #Neuron params
   V0::Float64 = -65.0
   Ca0::Float64 = 0.5

   # Sodium channel
   Na_g::Float64 = 100.0
   Na_E::Float64 = 50.0
   
   # Calcium-activated potassium channel
   KCa_g::Float64 = 5.0
   KCa_E::Float64 = -80.0
   
   # Slow calcium channel
   CaS_g::Float64 = 6.0
   CaS_E::Float64 = 0.0
   
   # Transient calcium channel
   CaT_g::Float64 = 2.5
   CaT_E::Float64 = 0.0
   
   # Potassium channel
   K_g::Float64 = 50.0
   K_E::Float64 = -80.0
   
   # Delayed rectifier potassium channel
   DRK_g::Float64 = 100.0
   DRK_E::Float64 = -80.0
   
   # Hyperpolarization-activated channel
   H_g::Float64 = 0.01
   H_E::Float64 = -20.0
   
   # Leak channel
   Leak_g::Float64 = 0.0
   Leak_E::Float64 = -50.0
   
   # Neuron capacitance
   C::Float64 = 1.0
end