
# Equations of unbalanced system currently
#
#
#
#
#
1. If channel reversal has a Ca Subsystem, then hook up to calcium concentration

2. If channel gate has a ca subsystem, then hook up to calcium flux 



  dmdt yes
 h infinity equation h∞(v,E) yes
 m infinity m∞(v,E) yes

 
Differential(t)(h(t)) ~ (h∞(t) - h(t)) / (60.0 + 150.0 / (exp(-0.0625(65.0 + E(t) + v(t))) + exp(0.1111111111111111(55.0 + E(t) + v(t)))))
 Differential(t)(m(t)) ~ (-m(t) + m∞(t)) / (1.4 + 7.0 / (exp(-0.07692307692307693(70.0 + E(t) + v(t))) + exp(0.1(27.0 + E(t) + v(t)))))
 Differential(t)(Ca(t)) ~ (Ca∞ - Ca(t) + (-flux_multiplier*ca₊n₊i(t)) / C) / τ
 Differential(t)(v(t)) ~ (sin(t) - g*(m(t)^3)*v(t)*h(t)) / C
 0 ~ -h∞(t) + 1.0 / (1.0 + exp(0.16129032258064516(60.0 + E(t) + v(t))))
 0 ~ 1.0 / (1.0 + exp(-0.1234567901234568(33.0 + E(t) + v(t)))) - m∞(t)



 dhdt(h, E, v, hinf)
 dmdt(m, minf, E,v)
 dCadt(Ca, ca.n.i)
 dvdt(v,m,h)
 hinf(E,v)
 minf(E,v)


 E is CaS.conductance.E which seems unset.
 :w
  
IMBALANCE:

julia> unknowns(neur)
9-element Vector{SymbolicUtils.BasicSymbolic{Real}}:
 CaS₊conductance₊h(t) - yes ODE
 CaS₊conductance₊m(t) - yes ODE 
 soma₊Ca(t) -yes ODE 
 soma₊v(t) -yes ODE
 CaS₊conductance₊v(t)
 CaS₊conductance₊E(t)
 CaS₊conductance₊m∞(t) - yes eqn f
 CaS₊conductance₊h∞(t) - yes eqn
 soma₊ca₊n₊i(t)

julia> equations(neur)
6-element Vector{Equation}:
 Differential(t)(CaS₊conductance₊h(t)) ~ (CaS₊conductance₊h∞(t) - CaS₊conductance₊h(t)) / CaS₊conductance₊τh(t)
 Differential(t)(CaS₊conductance₊m(t)) ~ (-CaS₊conductance₊m(t) + CaS₊conductance₊m∞(t)) / CaS₊conductance₊τm(t)
 Differential(t)(soma₊Ca(t)) ~ (soma₊Ca∞ - soma₊Ca(t) + (soma₊flux_multiplier*soma₊ca₊i(t)) / soma₊C) / soma₊τ
 Differential(t)(soma₊v(t)) ~ (soma₊I₊u(t) + soma₊i(t)) / soma₊C
 0 ~ -CaS₊conductance₊h∞(t) + 1.0 / (1.0 + exp(0.16129032258064516(60.0 + CaS₊conductance₊E(t) + CaS₊conductance₊v(t))))
 0 ~ 1.0 / (1.0 + exp(-0.1234567901234568(33.0 + CaS₊conductance₊E(t) + CaS₊conductance₊v(t)))) - CaS₊conductance₊m∞(t)






function build_channel(gate, Reversal)
    return @mtkmodel Channel begin
        @parameters begin
            g = g, [description = "Channel conductance"]
            E = E, [description = "Reversal Potential"]
        end
        @components begin
            p = Pin()
            n = Pin()
            reversal = Reversal(;V = E)
            conductance = gate(g=g, E=E)
        end
        @equations begin
            connect(conductance.p, reversal.n)
            connect(conductance.n, n)
            connect(reversal.p, p)
        end
    end
end



