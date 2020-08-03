using SafeTestsets

@time begin
    @time @safetestset "Initial" begin include("test.jl") end
end