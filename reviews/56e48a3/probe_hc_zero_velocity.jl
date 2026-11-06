# HyperCloud test/sims.jl:94-98 in a fresh session: zero velocity, "the run reaches tmax without a warning"
using HyperCloud, HyperCloudCore, Test
println("HyperCloudCore from ", pathof(HyperCloudCore))
UPWIND1 = Dict{Symbol, Any}(:Time_stepper => :RK2, :Scheme_name => :Upwind, :Scheme_order => 1, :Scheme_MLS_order => 1,
    :Scheme_upwind_alg_nd => :Classic, :Flux_name => :Upwind, :MOOD_criterion => :none)
linear_1d_gauss(N; tmax = 0.1, width = 0.1) = Dict{Symbol, Any}(
    :PDE_name => :linear, :PDE_velocities => ((1.0,),), :sim_func_name => :run_direct_simulation,
    :tmax => tmax, :snapshots => 2, :Grid_domain => :rectangular, :Grid_mins => (0.0,), :Grid_maxs => (1.0,), :Grid_Ns => (N,),
    :Grid_periodic => true, :Grid_randomness_factor => (0.0,), :Grid_seed => 42, :Grid_mover => :none,
    :Weight_name => :exponential, :Weight_alpha => 1.0, :Weight_range => 5.5,
    :IC_name => :gauss, :IC_a => (1.0,), :IC_b => (0.5,), :IC_width => width, :Time_CFL => 0.2)
pz = merge(linear_1d_gauss(50), UPWIND1, Dict{Symbol, Any}(:PDE_velocities => ((0.0,),)))
logs, simz = Test.collect_test_logs(() -> run_direct_simulation(pz))
println("reached tmax: ", simz !== nothing && simz.t[end] ≈ 0.1)
for l in logs; l.level >= Base.CoreLogging.Warn && println("WARNING LOGGED: ", l.message); end
println("HyperCloud assertion !any(level >= Warn) holds: ", !any(l -> l.level >= Base.CoreLogging.Warn, logs))
