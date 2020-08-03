using InverseGaussTransform

# ==========================
# TESTS
# ==========================

function trap_rule_test()
    Narr = [20,40,80,160,320,320*2,320*4,320*8,320*16,320*32]
    exact = 4^3/3
    errorArr = zeros(length(Narr))
    for ti=1:length(Narr)
        xarr = collect(range(0,stop=4.0,length=Narr[ti]))
        farr = xarr.^2
        quad = trap_rule(xarr,farr)
        errorArr[ti] = abs((quad - exact)/exact)
    end

    for tj=2:length(Narr)
        println("ratio: ",errorArr[tj-1]/errorArr[tj])
    end
    nothing
end

function cost_test()
    x1 = 0.7
    s = 1.2
    h = 0.5

    N = 200
    y = collect(range(-2,stop=2,length=N))
    F = zeros(N)
    for ti=1:N
        F[ti] = s*GaussKernel(y[ti] - x1,h)
    end

    # initial conditions [x0,s0,h0]
    #θ = [x1,s,h]
    #θ = [-1.0,1.0,0.5,0.2,0.2]
    θ = [-1.0,0.5,0.1]

    plot_fields(θ,y,F)
    J = cost(θ,y,F)

    nothing
end

function grad_cost_test()
    x1 = 0.7
    s = 1.2
    h = 0.5

    N = 200
    y = collect(range(-2,stop=2,length=N))
    F = zeros(N)
    for ti=1:N
        F[ti] = s*GaussKernel(y[ti] - x1,h)
    end

    # initial conditions [x0,s0,h0]
    #θ = [x1,s,h]
    #θ = [-1.0,1.0,0.5,0.2,0.2]
    θ = [0.0,0.0,0.1]

    plot_fields(θ,y,F)
    ∇J = grad_cost_FD(θ,y,F)
    println(∇J)

    nothing
end

function gradient_descent_test()
    #x1 = [0.7,-0.2]
    #s = [1.2,0.4]
    #x1 = 0.7
    #s  = 1.2
    #h = 0.5
    M = 100
    x1 = 2*rand(M) .- 1.0
    s1  = rand(M)
    h = 0.5

    N = 200
    y = collect(range(-2,stop=2,length=N))
    F = zeros(N)
    for ti=1:N
        for tj = 1:M
            F[ti] += s[tj]*GaussKernel(y[ti] - x1[tj],h)
        end
    end

    # initial conditions [x0,s0,h0]
    #θ = [x1,s,h]
    θ0 = [-0.3,0.0,0.1]
    #θ0 = [0.0,0.0,0.1]
    #θ0 = #[-1.0,1.0,0.5,0.2,0.2]
    #θ0 = [0.0,0.5,0.1]

    θ = gradient_descent(θ0,y,F;NiterMAX=1000,TOL=1e-8,pause=true)

    println(θ)
end

function gradient_descent_test_print_for_dissertation()
    M = 5
    x1 = 2*rand(M) .- 1.0
    s1  = rand(M)
    h = 0.5

    N = 200
    y = collect(range(-3,stop=3,length=N))
    F = zeros(N)
    for ti=1:N
        for tj = 1:M
            F[ti] += s1[tj]*GaussKernel(y[ti] - x1[tj],h)
        end
    end

    #θ0 = [-0.3,0.0,0.1]
    K= M+20
    θ0 = vcat(collect(range(-1,stop=1,length=K)),0.01*ones(K),0.1)

    error = 100.0
    α = 0.04

    p = plot(y,F,label="true",xlabel="y",xaxis=[-2,2])

    iter = 1
    θ = θ0
    Jtemp = 100.0
    NiterMAX = 2000
    TOL = 1e-8
    while iter<NiterMAX && error > TOL
        println("\riteration ",iter,"          ")        
        plot_fields(θ,y,F;iter=iter)

        #readline()

        J = cost(θ,y,F)
        error = abs(J - Jtemp)

        ∇J = grad_cost(θ,y,F)
        
        gradient_descent_step!(θ,∇J,α)

        G = get_gauss_field(θ,y)
        if mod(iter,40) == 0
            plot!(p,y,G,linestyle=:dash,label="iter $(iter)")
        end

        Jtemp = J
        iter += 1
    end

   savefig(p,"test.png")

    println(θ)
end

function gradient_descent_test_print_for_dissertation_COST()
    x1 = 0.7
    s1  = 1.2
    h = 0.5

    N = 200
    y = collect(range(-3,stop=3,length=N))
    F = zeros(N)
    for ti=1:N
        F[ti] += s1*GaussKernel(y[ti] - x1,h)
    end

    θ0 = [-0.3,0.0,0.1]

    error = 100.0
    α = 0.2

    Jarr = [cost(θ0,y,F)]

    iter = 1
    θ = θ0
    Jtemp = 100.0
    NiterMAX = 5000
    TOL = 1e-8
    while iter<NiterMAX && error > TOL
        println("\riteration ",iter,"          ")        
        #plot_fields(θ,y,F;iter=iter)

        J = cost(θ,y,F)
        push!(Jarr,J) 
        error = abs(J - Jtemp)

        ∇J = grad_cost(θ,y,F)
        
        gradient_descent_step!(θ,∇J,α)

        G = get_gauss_field(θ,y)

        Jtemp = J
        iter += 1
    end

    p = plot(0:(length(Jarr)-1),Jarr,ylabel="J (cost)",label="",xlabel="iteration #")

    savefig(p,"cost.png")

    println(θ)
end

function dissertation_STEP()
    N_pts = 200
    y = collect(range(-3,stop=3,length=N_pts))
    F = zeros(N_pts)
    for ti=1:N_pts
        F[ti] = step_function(y[ti])
    end

    iterArr = Array{Int}(undef,0)
    indArr  = Array{Int}(undef,0)
    minJArr = Array{Float64}(undef,0)

    for tj=19:20
        println("\r calculating N=$(tj)        ")
        push!(indArr,tj)
        θ0 = vcat(collect(range(-1,stop=1,length=tj)),0.01*ones(tj),0.1)

        error = 100.0
        α = 0.04

        Jarr = [cost(θ0,y,F)]
        J = zero(Float64)

        iter = 1
        θ = θ0
        Jtemp = 100.0
        NiterMAX = 10000
        TOL = 1e-16
        while iter<NiterMAX && error > TOL
            #println("\riteration ",iter,"          ")        
            #plot_fields(θ,y,F;iter=iter)

            J = cost(θ,y,F)
            #println(J)
            #push!(Jarr,J) 
            error = abs(J - Jtemp)

            ∇J = grad_cost(θ,y,F)
            
            gradient_descent_step!(θ,∇J,α)

            G = get_gauss_field(θ,y)

            Jtemp = J
            iter += 1
        end
        plot_fields(θ,y,F)
        plot(p,"test.png")
        readline()
        push!(iterArr,iter-1)
        push!(minJArr,J)
    end

    #p1 = plot(indArr,minJArr,ylabel="minimum J",label="",xlabel="M")
    #savefig(p1,"cost_multiN_J.png")
    #p2 = plot(indArr,iterArr,ylabel="iterations to convergence",label="",xlabel="M")
    #savefig(p2,"cost_multiN_iter.png")
    nothing
end

function step_function(x)
    if x > -0.5 && x < 0.5
        return 1.0
    else
        return 0.0
    end
end