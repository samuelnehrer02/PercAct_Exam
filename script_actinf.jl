using Pkg

#Pkg.activate("path\\to\\local\\environment")
Pkg.add(url="https://github.com/samuelnehrer02/ActiveInference.jl.git")

### Loading packages
using Plots
using LinearAlgebra
using ActiveInference
using ActiveInference.Environments

#====================================== 🌍 Create GridWorld 🌍 ======================================# 
grid_locations = collect(Iterators.product(1:5, 1:7))
grid_dims = size(grid_locations)
# Visualize Grid Locations
plot_gridworld(grid_locations)

n_grid_points = prod(grid_dims)
location_to_index = Dict(loc => idx for (idx, loc) in enumerate(grid_locations))

cue1_location = (3,1)

cue2_loc_names = ["L1","L2","L3","L4"]
cue2_locations = [(1, 3), (2, 4), (4, 4), (5, 3)]

reward_conditions = ["TOP", "BOTTOM"]
reward_locations = [(2,6), (4,6)]
#====================================== 🧠 Defining the Generative Model 🧠 ======================================# 

# 3 state factors
n_states = [n_grid_points, length(cue2_locations), length(reward_conditions)]

# 4 observation modalities
cue1_names = ["Null";cue2_loc_names]
cue2_names = ["Null", "reward_on_top", "reward_on_bottom"]
reward_names = ["Null", "Cheese", "Shock"]
n_obs = [n_grid_points, length(cue1_names), length(cue2_names), length(reward_names)]

#------------------ A-matrix ------------------
A_m_shapes = [[o_dim; n_states] for o_dim in n_obs]
A = array_of_any_zeros(A_m_shapes)

# 35x35 identity matrix 
identity_matrix = Matrix{Float64}(I, n_grid_points,n_grid_points)
# Keep it as 35x35 identity matrix but adds two more dimensions :,:,1,1 
expanded_matrix = reshape(identity_matrix, size(identity_matrix, 1), size(identity_matrix, 2), 1, 1)
# Take expanded_matrix and makes multiple copies of it along the last two dimensions.
tiled_matrix = repeat(expanded_matrix, outer=(1, 1, n_states[2], n_states[3]))
A[1] = tiled_matrix

A[2][1,:,:,:] .= 1.0

for (i, cue_loc2_i) in enumerate(cue2_locations)
    A[2][1, location_to_index[cue1_location], i, :] .= 0.0
    A[2][i+1, location_to_index[cue1_location], i, :] .= 1.0
end 

# A[3] = Third Modality = Cue-2 Observation Modality = [3, 35, 4, 2]
A[3][1,:,:,:] .= 1.0 # NULL is the most likely observation everywhere

for (i, cue_loc2_i) in enumerate(cue2_locations)
    A[3][1,location_to_index[cue_loc2_i],i,:] .= 0.0
    A[3][2,location_to_index[cue_loc2_i],i,1]  = 1.0
    A[3][3,location_to_index[cue_loc2_i],i,2]  = 1.0

end

# A[4] = Fourth Modality = Reward Observation Modality = [3, 35, 4, 2]
A[4][1,:,:,:] .= 1.0 # NULL is the most likely observation everywhere

rew_top_idx = location_to_index[reward_locations[1]]
rew_bott_idx = location_to_index[reward_locations[2]]

# Agent is in the TOP reward condition
A[4][1,rew_top_idx,:,:] .= 0.0
A[4][2,rew_top_idx,:,1] .= 1.0
A[4][3,rew_top_idx,:,2] .= 1.0

# Agent is in the BOTTOM reward condition
A[4][1,rew_bott_idx,:,:] .= 0.0
A[4][2,rew_bott_idx,:,2] .= 1.0
A[4][3,rew_bott_idx,:,1] .= 1.0 

#------------------ B-matrix ------------------
n_controls = [5, 1, 1]
B_f_shapes = [[ns, ns, n_controls[f]] for (f, ns) in enumerate(n_states)]
B = array_of_any_zeros(B_f_shapes)
# actions vector
actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
len_y, len_x = size(grid_locations)

# Populating the first dimension of the B-Matrix
for (action_id, action_label) in enumerate(actions)

    for (curr_state, grid_locations) in enumerate(grid_locations)

        y, x = grid_locations

        next_y, next_x = y, x
        if action_label == "DOWN"
            next_y = y < len_y ? y + 1 : y
        elseif action_label == "UP"
            next_y = y > 1 ? y - 1 : y
        elseif action_label == "LEFT"
            next_x = x > 1 ? x - 1 : x
        elseif action_label == "RIGHT"
            next_x = x < len_x ? x + 1 : x
        elseif action_label == "STAY"    
        end

        new_location = (next_y, next_x)
        next_state = location_to_index[new_location]

        B[1][next_state, curr_state, action_id] = 1.0

    end

end

# Populating the 2nd and 3rd layer of the B-matrix with identity-matrices
B[2][:,:,1] = Matrix{Float64}(I, n_states[2], n_states[2])
B[3][:,:,1] = Matrix{Float64}(I, n_states[3], n_states[3])

#------------------ C-vectors ------------------
C = array_of_any_zeros(n_obs)
C[4][2] = 2.0
C[4][3] = -4.0
#------------------ D-vectors ------------------
D = array_of_any_uniform(n_states) 
D[1] = onehot(location_to_index[(1, 1)], n_grid_points)


#====================================== 💻 Simulation and Data Collection 💻 ======================================# 

#= !!!  Data from simulations were saved and can be loaded from DataFrames lower in the script (line: 227) !!! =#

# Empty vectors for the collected data
collected_observations = []
action_ids = []
collected_q_pi = []

# Locations used in simulations
cue1_location = (3, 1)
starting_locations = [(1, 1), (5, 1), (3, 4)]
initial_observations = [[1, 1, 1, 1], [5, 1, 1, 1], [18, 1, 1, 1]]  

# Total number of timesteps for each simulation
T = 12
# Run simulation
for cue2_loc in cue2_loc_names
    for reward_cond in reward_conditions
        for (start_loc, initial_obs) in zip(starting_locations, initial_observations)
            println("Simulation at Cue2 Location: $(cue2_loc), Reward Condition: $(reward_cond), Start Location: $(start_loc)")

            # Initializing the environment and agent for each new combination
            D[1] = onehot(location_to_index[start_loc], n_grid_points)
            my_agent = initialize_agent(A, B, C, D, policy_len = 4, alpha=4.0) ### Change the alpha parameter to desired value!
            env = EpistChainEnv(start_loc, cue1_location, cue2_loc, reward_cond, grid_locations)
            obs = initial_obs

            # Run the trial
            for t in 1:T
                println("Time t ------------------------------ $(t) ------------------------------")
                println("Agent at time: $(t) Observes itself at: $(obs)")
                qs = infer_states!(my_agent, obs)

                q_pi, G = infer_policies!(my_agent)
                push!(collected_q_pi, q_pi)

                chosen_action_id = sample_action!(my_agent)
                println("Chosen Action $(chosen_action_id)")
                push!(action_ids, (chosen_action_id))

                movement_id = Int(chosen_action_id[1])
                choice_action = actions[movement_id]
                println("Action Taken: $(choice_action)")

                loc_obs, cue1_obs, cue2_obs, reward_obs = step!(env, choice_action)
                obs = [location_to_index[loc_obs], findfirst(isequal(cue1_obs), cue1_names), findfirst(isequal(cue2_obs), cue2_names), findfirst(isequal(reward_obs), reward_names)]
                println("Observation after taking action: $(loc_obs), $(cue1_obs), $(cue2_obs), $(reward_obs)")
                push!(collected_observations, (obs))

                println("Reward at time $(t): $(reward_obs)")
            end
        end
    end
end

#====================== Saving the data as a dataframes ==========================#
Pkg.add("DataFrames")
using Serialization
using DataFrames

simulation_alpha_16 = DataFrame(
    Observations = collected_observations,
    Actions = action_ids,
    Q_Pi = collected_q_pi
)

serialize("simulation_alpha_16.jls", simulation_alpha_16)
#-------------------------------------------------------------------------------#
simulation_alpha_4 = DataFrame(
    Observations = collected_observations,
    Actions = action_ids,
    Q_Pi = collected_q_pi
)

serialize("simulation_alpha_4.jls", simulation_alpha_4)
#-------------------------------------------------------------------------------#
simulation_alpha_25 = DataFrame(
    Observations = collected_observations,
    Actions = action_ids,
    Q_Pi = collected_q_pi
)

serialize("simulation_alpha_25.jls", simulation_alpha_25)

#====================================== 🔬 Esimate Alpha 🔬 ======================================#
using Turing
using StatsPlots
using Random
using Distributions
using MCMCChains
using DynamicPPL: getlogp, getval
using Optim

####  Loading the data from the data frames #### 

# Transforming Posteriors over policies to a matrix
# Transforming action vectors to integers 
subject_data = deserialize("subject_data.jls")
Q_Pi_Real = hcat(Array(subject_data[!,:Q_Pi])...)
Actions_Real = subject_data[!,:Actions]

simulation_alpha_4 = deserialize("simulation_alpha_4.jls")
Q_Pi_4 = hcat(Array(simulation_alpha_4[!,:Q_Pi])...)
Actions_4 = Int64[first(action) for action in simulation_alpha_4.Actions]

simulation_alpha_16 = deserialize("simulation_alpha_16.jls")
Q_Pi_16 = hcat(Array(simulation_alpha_16[!,:Q_Pi])...)
Actions_16 = Int64[first(action) for action in simulation_alpha_16.Actions]

simulation_alpha_25 = deserialize("simulation_alpha_25.jls")
Q_Pi_25 = hcat(Array(simulation_alpha_25[!,:Q_Pi])...)
Actions_25 = Int64[first(action) for action in simulation_alpha_25.Actions]

### Custom function for computing the action marginals (not a part of the package yet, needs to be made generic)
function get_log_marginal_f(Q_pi, policies, num_controls)
    num_timesteps = size(Q_pi, 2)
    num_factors = length(num_controls)
    log_marginal_f_storage = Array{Array{Float64, 1}, 1}(undef, num_timesteps)

    for t in 1:num_timesteps
        action_marginals = array_of_any_zeros(num_controls)
        for (pol_idx, policy) in enumerate(policies)
            for (factor_i, action_i) in enumerate(policy[1, :])
                if num_controls[factor_i] > 1
                    action_marginals[factor_i][action_i] += Q_pi[pol_idx, t]
                end
            end
        end
        action_marginals = norm_dist_array(action_marginals)
        log_marginal_f_timestep = []
        for factor_i in 1:num_factors
            if num_controls[factor_i] > 1
                log_marginal_f = spm_log_single(action_marginals[factor_i])
                append!(log_marginal_f_timestep, log_marginal_f)
            end
        end
        log_marginal_f_storage[t] = log_marginal_f_timestep
    end
    return log_marginal_f_storage
end

### Applying the function to calculate the action marginals, transforming them into matrices
# initialize agent to retrieve policies and num_controls
my_agent = initialize_agent(A, B, C, D, policy_len = 4)
policies = my_agent.policies
num_controls = my_agent.num_controls

log_marginal_f_real = hcat(get_log_marginal_f(Q_Pi_Real, policies, num_controls)...)
log_marginal_f_4 = hcat(get_log_marginal_f(Q_Pi_4, policies, num_controls)...)
log_marginal_f_16 = hcat(get_log_marginal_f(Q_Pi_16, policies, num_controls)...)
log_marginal_f_25 = hcat(get_log_marginal_f(Q_Pi_25, policies, num_controls)...)

#================================ *** Turing.jl Probabilistic model ***  ===============================#


@model function estimate_alpha(log_marginal_f_matrix, Action_sub_int)
    α ~ Gamma(4, 5)

    for t in eachindex(Action_sub_int)

        log_marginal_f = log_marginal_f_matrix[:, t]

        action_probs = softmax(log_marginal_f * α)
        
        Action_sub_int[t] ~ Categorical(action_probs)
    end
end

### !!! Chains were saved and can be loaded on line: 317 !!!

### Run NUTS sampler and save the chains
#=
chain_4 = sample(estimate_alpha(log_marginal_f_4, Actions_4), NUTS(), 3000)
chain_16 = sample(estimate_alpha(log_marginal_f_16, Actions_16), NUTS(), 3000)
chain_25 = sample(estimate_alpha(log_marginal_f_25, Actions_25), NUTS(), 3000)
chain_real = sample(estimate_alpha(log_marginal_f_real, Actions_Real), NUTS(), 3000)

serialize("chain_4.bin", chain_4)
serialize("chain_16.bin", chain_16)
serialize("chain_25.bin", chain_25)
serialize("chain_real.bin", chain_real)
=#

# Load saved chains
chain_4 = deserialize("chain_4.bin")
chain_16 = deserialize("chain_16.bin")
chain_25 = deserialize("chain_25.bin")
chain_real = deserialize("chain_real.bin")

#==================================== Posterior Diagnostics ===================================#
summary_chain = describe(chain_real)
display(summary_chain[1])
display(summary_chain[2])

chain_4_p = plot(chain_4,  title="Sim1: Alpha = 4.0", xlabel = false, ylabel = false, legend = false)
chain_16_p = plot(chain_16, title="Sim2: Alpha = 16.0", xlabel = false, ylabel = false, legend = false)
chain_25_p = plot(chain_25, title="Sim3: Alpha = 25.0",xlabel = false, ylabel = false, legend = false)
chain_real_p = plot(chain_real, title="Human Subject" ,xlabel = false, ylabel = false, legend = false)
combined_trace_plots = plot(chain_4_p, chain_16_p, chain_25_p, chain_real_p, layout = (4,1), size=(700,1000))

#================================= *** MAP Estimation *** ================================#
# Turing.jl probabilistic model adapted for optimization
@model function optimize_alpha(log_marginal_f_matrix, Action_sub_int, α)
    α ~ Gamma(4, 5)

    for t in eachindex(Action_sub_int)

        log_marginal_f = log_marginal_f_matrix[:, t]

        action_probs = softmax(log_marginal_f .* α)
        
        Action_sub_int[t] ~ Categorical(action_probs)
    end
end

# Defining objective function: Retrieves the negative log posterior from the Turing model
function negative_log_posterior(α, log_marginal_f_matrix, Actions)
    model_instance = optimize_alpha(log_marginal_f_matrix, Actions, α)
    
    var_info = Turing.VarInfo(model_instance)
    model_instance(var_info)
    return -getlogp(var_info)
end

# Optimization via Optim's Newton method with optimized tolerance for better convergence 
opt_result_4 = optimize(α -> negative_log_posterior(α, log_marginal_f_4, Actions_4), [1.0], Newton(), Optim.Options(g_tol = 1e-6, x_tol = 1e-6, f_tol = 1e-6))
alpha_map_4 = Optim.minimizer(opt_result_4)

opt_result_16 = optimize(α -> negative_log_posterior(α, log_marginal_f_16, Actions_16), [1.0], Newton(), Optim.Options(g_tol = 1e-6, x_tol = 1e-6, f_tol = 1e-6))
alpha_map_16 = Optim.minimizer(opt_result_16)

opt_result_25 = optimize(α -> negative_log_posterior(α, log_marginal_f_25, Actions_25), [1.0], Newton(), Optim.Options(g_tol = 1e-6, x_tol = 1e-6, f_tol = 1e-6))
alpha_map_25 = Optim.minimizer(opt_result_25)

opt_result_real = optimize(α -> negative_log_posterior(α, log_marginal_f_real, Actions_Real), [1.0], Newton(), Optim.Options(g_tol = 1e-6, x_tol = 1e-6, f_tol = 1e-6))
alpha_map_real = Optim.minimizer(opt_result_real)

#================================= ✨ Plots used in the paper ✨ ================================#


# Collect samples from prior and posterior distributions
alpha_samples_4 = Array(chain_4[:α])
prior_4 = sample(estimate_alpha(log_marginal_f_4, Actions_4), Prior(), 3000)
prior_4_samples = Array(prior_4[:α])
alpha_samples_16 = Array(chain_16[:α])
prior_16 = sample(estimate_alpha(log_marginal_f_16, Actions_16), Prior(), 3000)
prior_16_samples = Array(prior_16[:α])
alpha_samples_25 = Array(chain_25[:α])
prior_25 = sample(estimate_alpha(log_marginal_f_25, Actions_25), Prior(), 3000)
prior_25_samples = Array(prior_25[:α])
alpha_samples_real = Array(chain_real[:α])
prior_real = sample(estimate_alpha(log_marginal_f_real, Actions_Real), Prior(), 3000)
prior_real_samples = Array(prior_real[:α])


##### Plotting Densities #####
plot_layout = @layout([A B; C D])
p = plot(layout=plot_layout, size=(1100, 600))

density!(p[1], alpha_samples_4, label="Posterior", fill=(0, 0.6, :springgreen3),xlims=(0,60), color=:springgreen3,
             title="Sim 1: Generative Alpha α = 4.0", xlabel="", ylabel="", linewidth=0.8, legend=:topright)
density!(p[1], prior_4_samples, label="Prior", fill=(0, 0.35, :indigo), xlims=(0,60),  color=:indigo, linewidth=0.8)
density!(p[2], alpha_samples_16, label="Posterior", fill=(0, 0.6, :springgreen3),xlims=(0,60), color=:springgreen3,
             title="Sim 2: Generative Alpha α = 16.0", xlabel="", ylabel="", linewidth=0.8, legend=:topright)
density!(p[2], prior_16_samples, label="Prior", fill=(0, 0.35, :indigo), xlims=(0,60),  color=:indigo, linewidth=0.8)
density!(p[3], alpha_samples_25, label="Posterior", fill=(0, 0.6, :springgreen3),xlims=(0,60), color=:springgreen3,
             title="Sim 3: Generative Alpha α = 25.0", xlabel="", ylabel="", linewidth=0.8, legend=:topright)
density!(p[3], prior_25_samples, label="Prior", fill=(0, 0.35, :indigo), xlims=(0,60),  color=:indigo, linewidth=0.8)
density!(p[4], alpha_samples_real, label="Posterior", fill=(0, 0.6, :springgreen3),xlims=(0,60), color=:springgreen3,
             title="Estimated Alpha - Human Subject", xlabel="", ylabel="", linewidth=0.8, legend=:topright)
density!(p[4], prior_real_samples, label="Prior", fill=(0, 0.35, :indigo), xlims=(0,60),  color=:indigo, linewidth=0.8)

##### Plotting MAP's #####
generative_alpha = [4.0, 16.0, 25.0]
estimated_alpha = [alpha_map_4, alpha_map_16, alpha_map_25]
estimated_alpha = map(x -> x[1], estimated_alpha)
# Regression line
scatter(generative_alpha, estimated_alpha, title="Simulations: Generative vs Estimated Alpha",
        smooth=true, xlabel="Generative Alpha", ylabel="Estimated Alpha", legend=false,
        xlims=(0, 30),ylims=(0,25) ,color=:purple,
        xticks=0:5:30, yticks=0:5:25, size=(750, 500))
cor_coefficient = cor(generative_alpha, estimated_alpha)
annotate!(mean(generative_alpha), mean(estimated_alpha), text("r = $(round(cor_coefficient, digits=6))", :left, :top))


##### Barplots #####
estimated_parameters = [alpha_map_4, alpha_map_16, alpha_map_25, alpha_map_real]
estimated_parameters = map(x -> x[1], estimated_parameters)
prior_means = [20,20,20,20] 
categories = ["Simulation 1","Simulation 2","Simulation 3","Human Subject"] 
bar(categories, prior_means, label="Prior Mean", color=:blue3,linecolor=:blue3, alpha=0.7, bar_width=0.75, ylims=(0,35), yticks=0:2.5:35)
bar!(categories, estimated_parameters, label="Estimated Parameters", linecolor=:brown1 , linewidth=0.5, color=:brown1, alpha=0.8, bar_width=0.5)
scatter!(categories[1:3], generative_alpha, label="Generative Alpha",linewidth =2, markershape = :cross, color=:black, markersize = 6, markerstrokewidth = 3)
title!("(α) Prior Means and Estimated Alpha", position=:left)
plot!()
