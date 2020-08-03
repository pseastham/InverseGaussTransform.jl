function trap_rule(xarr::Vector{T},farr::Vector{T}) where T
    N = length(xarr)

    Δx = xarr[2] - xarr[1]
    quad = zero(T)
    for ti=1:N-1
        quad += farr[ti] + farr[ti+1]
    end

    quad *= 0.5*Δx

    return quad
end

# return J. Compares how well particle field (represented by θ) 
# approximates F(y)
function cost(θ,y,F)
    x,s,h = decompose_theta(θ)
    G = get_gauss_field(θ,y)
    f = (G - F).^2
    J = trap_rule(y,f)
    return J
end

function grad_cost(θ,y,F)
    dJdx = compute_dJdx(θ,y,F)
    dJds = compute_dJds(θ,y,F)
    dJdh = compute_dJdh(θ,y,F)
    dJdθ = vcat(dJdx,dJds,dJdh)

    return dJdθ
end

function compute_dJdx(θ,y,F)
    x,s,h = decompose_theta(θ)
    G = get_gauss_field(θ,y)

    N = length(x)
    M = length(y)
    dJdx = zeros(N)

    for ti=1:N
        f = zeros(M)
        for tj=1:M
            f[tj] = 2*(G[tj] - F[tj])*compute_dGdx(x[ti],s[ti],h,y[tj])
        end
        dJdx[ti] = trap_rule(y,f)
    end
    return dJdx
end

function compute_dJds(θ,y,F)
    x,s,h = decompose_theta(θ)
    G = get_gauss_field(θ,y)

    N = length(x)
    M = length(y)
    dJds = zeros(N)

    for ti=1:N
        f = zeros(M)
        for tj=1:M
            f[tj] = 2*(G[tj] - F[tj])*compute_dGds(x[ti],s[ti],h,y[tj])
        end
        dJds[ti] = trap_rule(y,f)
    end
    return dJds
end

function compute_dJdh(θ,y,F)
    x,s,h = decompose_theta(θ)
    G = get_gauss_field(θ,y)

    M = length(y)
    f = zeros(M)
    for tj=1:M
        f[tj] = 2*(G[tj] - F[tj])*compute_dGdh(x,s,h,y[tj])
    end
    dJdh = trap_rule(y,f)
    return dJdh
end

function compute_dGdx(x::T,s::T,h::T,y::T) where T
    dGdx = -2*s*(x - y)*GaussKernel(x-y,h)/h^2
    return dGdx
end

function compute_dGds(x::T,s::T,h::T,y::T) where T
    dGds = GaussKernel(x-y,h)
    return dGds
end

function compute_dGdh(x::Vector{T},s::Vector{T},h::T,y::T) where T
    N = length(x)
    dGdh = zero(T)
    for ti=1:N
        dGdh += s[ti]*(x[ti] - y)^2*GaussKernel(x[ti]-y,h)
    end
    dGdh *= 2/h^3
    return dGdh
end

# computes grad_cost with finite difference method
# to brute-force check the analytic gradient
function grad_cost_FD(θ,y,F)
    n_temp = length(θ)
    N = Int((n_temp-1)/2)

    Δx = 0.01*(maximum(y) -minimum(y))
    Δs = 0.01*maximum(F)
    Δh = Δx
    dJdx = zeros(N)
    dJds = zeros(N)
    dJdh = 0.0
    
    # compute dJdx
    for ti=1:N
        θtemp = copy(θ)
        θtemp[ti] = θtemp[ti] + Δx
        J1 = cost(θtemp,y,F)
        J0 = cost(θ,y,F)
        dJdx[ti] = (J1 - J0)/Δx
    end

    # compute dJds
    for ti=1:N
        θtemp = copy(θ)
        θtemp[ti+N] = θtemp[ti+N] + Δs
        J1 = cost(θtemp,y,F)
        J0 = cost(θ,y,F)
        dJds[ti] = (J1 - J0)/Δs
    end

    # compute dJdh
    θtemp = copy(θ)
    θtemp[2N+1] = θtemp[2N+1] + Δh
    J1 = cost(θtemp,y,F)
    J0 = cost(θ,y,F)
    dJdh = (J1 - J0)/Δh

    dJdθ = vcat(dJdx,dJds,dJdh)
    return dJdθ
end

function plot_fields(θ,y,F;iter=-1)
    x,s,h = decompose_theta(θ)

    G = get_gauss_field(θ,y)
    p = plot()
    if iter == -1
        plot!(p,y,G,label="G")
    else
        plot!(p,y,G,label="G",title="iteration $(iter)")
    end
    plot!(p,y,F,label="F")
    gui(p)

    nothing
end

function get_gauss_field(θ,y)
    # decompose theta
    x,s,h = decompose_theta(θ)

    # plot gauss field from particles
    d = 1                                   # dimension of data
    M = length(y)                           # number of targets
    N = length(x)                           # number of sources
    ε = 1e-2                                # max tolerance
    W = 1                                   # number of sources being evaluated
    G = zeros(M)                            # output values array
    mydgt!(G,d,M,N,h,ε,x,y,s,W)
    return G
end

function gradient_descent_step!(θ,∇f,α)
    N = length(θ)
    for ti=1:N
        θ[ti] -= α*∇f[ti]
    end
    nothing
end

function gradient_descent(θ0,y,F;NiterMAX=100,TOL=1e-10,pause=false)
    error = 100.0
    α = 0.2

    iter = 1
    θ = θ0
    Jtemp = 100.0
    while iter<NiterMAX && error > TOL
        println("\riteration ",iter,"          ")        
        plot_fields(θ,y,F;iter=iter)

        J = cost(θ,y,F)
        error = abs(J - Jtemp)

        #∇J = grad_cost_FD(θ,y,F)
        ∇J = grad_cost(θ,y,F)
        
        gradient_descent_step!(θ,∇J,α)
                
        if pause; readline(); end

        Jtemp = J

        iter += 1
    end

    return θ
end

function compose_theta(x,s,h)
    N = length(x)
    θ = zeros(2N+1)
    for ti=1:N
        θ[ti] = x[ti]
        θ[ti+N] = s[ti]
    end
    θ[2N+1] = h
    return θ
end

function decompose_theta(θ)
    n_temp = length(θ)
    N = Int((n_temp-1)/2)
    x = zeros(N)
    s = zeros(N)
    for ti=1:N
        x[ti] = θ[ti]
        s[ti] = θ[ti+N]
    end
    h = θ[2N+1]
    return x,s,h 
end