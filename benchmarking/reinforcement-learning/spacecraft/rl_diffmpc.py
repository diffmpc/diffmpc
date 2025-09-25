"""RL training on the spacecraft problem."""
import jax.numpy as jnp
from jax import config, vmap, jit
import jax
import time
import argparse

# DiffMPC imports
config.update("jax_enable_x64", True) # use double precision
from diffmpc.dynamics.spacecraft_dynamics import SpacecraftDynamics
from utils import reward, generate_problem_data
from diffmpc.problems.optimal_control_problem import OptimalControlProblem
from diffmpc.solvers.sqp import SQPSolver
from diffmpc.utils.load_params import load_problem_params
from diffmpc.utils.load_params import load_solver_params
import matplotlib.pyplot as plt

def benchmark_diffmpc_rollout(num_batch, num_sim_steps, horizon):
    print("=== Running RL training, then comparing state trajectories ===")

    num_steps = 2000
    lr = 1.0

    # Problem parameters
    problem_params = load_problem_params("spacecraft.yaml")
    inertia_vector, initial_states = generate_problem_data(num_batch, seed=0)
    print(f"device: {inertia_vector.device}")
    diffmpc_horizon = horizon - 1  # diffmpc horizon is N+1
    problem_params["horizon"] = diffmpc_horizon
    problem_params["inertia_vector"] = inertia_vector
    problem_params["reference_state_trajectory"] = jnp.zeros((diffmpc_horizon + 1, 3))
    problem_params["reference_control_trajectory"] = jnp.zeros((diffmpc_horizon + 1, 3))
    problem_params["weights_penalization_final_state"] = jnp.zeros(3)

    # Solver parameters
    solver_params = load_solver_params("sqp.yaml")
    solver_params["num_scp_iteration_max"] = 1
    solver_params["pcg"]["tol_epsilon"] = 1.e-12
    solver_params["linesearch"] = True
    solver_params["linesearch_alphas"] = [1.0]

    dynamics = SpacecraftDynamics()
    problem = OptimalControlProblem(dynamics=dynamics, params=problem_params)
    solver = SQPSolver(program=problem, params=solver_params)

    def solver_initial_guess(initial_state):
        params = {**problem_params, "initial_state": initial_state}
        return solver.initial_guess(params)

    weights = {
        k: problem_params[k] for k in [
            "weights_penalization_reference_state_trajectory",
            "weights_penalization_control_squared"
        ]
    }

    problem_params["initial_state"] = initial_states[0]

    def rollout(state, weights):
        initial_carry = (state, solver_initial_guess(state), 0.0)

        def rollout_step(carry, _):
            state, solution, running_reward = carry
            
            # Solve MPC problem  
            params = {
                **problem_params,
                "inertia_vector": inertia_vector,
                "initial_state": state
            }
            solution = solver.solve_differentiable(solution, params, weights)
            control = solution.controls[0]

            # Simulate forward
            dt = params["discretization_resolution"]
            next_state = state + dt * dynamics.state_dot(state, control, params)
            running_reward = running_reward + reward(next_state, control)

            return (next_state, solution, running_reward), None

        final_carry, _ = jax.lax.scan(rollout_step, initial_carry, None, length=num_sim_steps)
        _, _, final_reward = final_carry
        return final_reward

    rollout_batch = vmap(rollout, in_axes=(0, None))

    def rollout_for_grad(weights):
        return jnp.sum(rollout_batch(initial_states, weights))
    print("initial_states =", initial_states)

    @jit
    def grad_step_and_loss(weights):
        loss, grad = jax.value_and_grad(rollout_for_grad)(weights)
        for key in weights.keys():
            weights[key] = jnp.maximum(1e-10, weights[key] + lr * jnp.nan_to_num(grad[key]))
        return weights, loss
    jax.block_until_ready(grad_step_and_loss(weights))

    # plot rollouts with base weights (pre-RL)
    solution = solver.solve_differentiable(
        solver.initial_guess(problem_params),
        problem_params,
        weights
    )
    fig, axes = plt.subplots(nrows = dynamics.num_states + dynamics.num_controls, figsize=(8, 12))
    for i, ax in enumerate(axes.flatten()):
        ax.plot(solution.states[:, i], 'b--')

    # RL training
    s = time.monotonic()
    for step in range(num_steps):
        weights, loss = grad_step_and_loss(weights)
        jax.block_until_ready((weights, loss))
        print("state cost weight =", weights["weights_penalization_reference_state_trajectory"])

        ts = time.monotonic() - s
        # print(f'Step {1+step}/{num_steps}, reward: {loss}, weight: {weights}, time: {ts}')
        print(f'Step {1+step}/{num_steps}, reward: {loss}, time: {ts}')
    print(f"Training elapsed ms {1000*ts/(num_batch*num_sim_steps*num_steps):.3f} per step/problem/trainingstep")

    # plot rollouts with learned weights (post-RL)
    solution = solver.solve_differentiable(
        solver.initial_guess(problem_params),
        problem_params,
        weights
    )
    for i, ax in enumerate(axes.flatten()):
        ax.plot(solution.states[:, i], 'b')
        ax.grid(True)

    plt.show()
    
    return None



def main():
    parser = argparse.ArgumentParser(description='MPC Benchmark')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size for benchmark')
    parser.add_argument('--horizon', type=int, default=40,
                       help='MPC horizon')
    parser.add_argument('--num_sim_steps', type=int, default=20,
                       help='Number of simulation steps')
    
    args = parser.parse_args()
    
    print(f"\n{'=' * 70}")
    print(f"DiffMPC Rollout: {args.horizon} horizon")
    print(f"Simulation: {args.num_sim_steps} steps, {args.batch_size} batch")
    
    benchmark_diffmpc_rollout(args.batch_size, args.num_sim_steps, args.horizon)

if __name__ == "__main__":
    main()
