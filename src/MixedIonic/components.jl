@connector IonicPin begin
    q(t)                  # Concentration at the pin [V]
    i(t), [connect = Flow]    # Current flowing into the pin [A]
end

@mtkmodel IonicPort begin       #Replicates oneport structure for concentrations
    @components begin
        p = IonicPin()
        n = IonicPin()
    end
    @variables begin
        q(t)                    #Bidirectional, for listening to calcium qs and pushing to calcium qs
        i(t)
    end
    @equations begin
        q ~ p.q - n.q           #
        0 ~ p.i + n.i
        i ~ p.i
    end
end

@mtkmodel IonicTerminal begin   #Same thing here
    @components begin
        p = IonicPin()
        n = IonicPin()
    end
    @variables begin
        q(t)                    #Monodirectional, only for listening to calcium qs
        i(t)
    end
    @equations begin
        q ~ p.q 
        n.q ~ 0
        n.i ~ 0
        i ~ p.i
        p.i ~ 0
    end
end

@mtkmodel IonicGround begin
    @components begin           #Connects to ionicterminal, enables no ca pushing.
        g = IonicPin()
    end
    @equations begin
        g.q ~ 0
    end
end

@mtkmodel DirectionalTwoPort begin
   @components begin
       pre = Pin()    # Presynaptic (voltage sensing)
       post = Pin()   # Postsynaptic (current injection)
   end
   @variables begin
       v_pre(t)
       v_post(t) 
       i_post(t)
   end
   @equations begin
       v_pre ~ pre.v
       v_post ~ post.v
       i_post ~ post.i
       
       # Directional constraint
       pre.i ~ 0  # No current drawn from presynaptic
   end
end

@mtkmodel BiDirectionalTwoPort begin
   @components begin
       pre = Pin() 
       post = Pin()
   end
   @variables begin
       v_pre(t)
       v_post(t) 
       i_post(t)
       i_pre(t)
   end
   @equations begin
       v_pre ~ pre.v
       v_post ~ post.v
       i_post ~ post.i
       i_pre ~ pre.i
   end
end
