using StokesParticles
using Plots

function gui_test()
    d = 1                      # dimension of data
    M = 10_000                 # number of targets 
    N = 2                      # number of sources
    h = 0.5                    # bandwidth
    ε = 1e-2                   # max tolerance
    #x = 2*rand(N*d) .- 1      # source vectors, reshape'd to 1D array
    x = [-1.0,1.0]
    y = collect(range(-2,stop=2,length=M))                   # target vectors, reshape'd to 1D array
    W = 1                           # number of sources being evaluated 
    q=[0.5,1.0]
    v1 = rand(M)                    # output values array
    v2 = rand(M)                    # output 2
    v3 = rand(M)                    # output 3

    @time StokesParticles.fgt!(v1,d,M,N,h,ε,x,y,q,W)
    @time StokesParticles.dgt!(v2,d,M,N,h,ε,x,y,q,W)
    @time StokesParticles.mydgt!(v3,d,M,N,h,ε,x,y,q,W)

    p1 = plot(y,v1,label="fgt (c++)")
    p2 = plot(y,v2,label="dgt (c++)")
    p3 = plot(y,v3,label="mydgt (julia)")

    l = @layout [a a a]
    p = plot(p1,p2,p3,layout = l)
    gui(p)
end

function 
