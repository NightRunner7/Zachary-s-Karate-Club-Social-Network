"""
Configuration Module for Network Evolution Simulation

This module, `configuration.py`, contains functions and utilities to set and adjust simulation parameters
for a network evolution model. The module is designed to provide a centralized location for managing simulation
settings, which can be imported and utilized across various parts of the project, ensuring consistency and
ease of adjustments.

The module's primary functionality includes defining base settings, scaling time and other parameters based on
the diffusion rate, and providing utility functions to ensure these parameters are correctly integrated into
the simulation environment. It supports the exploration of hypotheses related to the effective diffusion rate
(Deff = D/β) and its impact on the universality of network evolution dynamics.

Functions:
    - adjust_time_for_diffusion(sim_config, diffusion, base_dt, base_time_end, check_interval, draw_interval, update_interval):
      Scales time-related simulation parameters based on the diffusion rate. It is used to maintain consistent
      simulation dynamics, particularly focusing on how diffusion affects state propagation speeds.

    - adjust_time_for_diffusion_vol2(sim_config, diffusion, base_dt, base_time_end, check_interval, draw_interval, update_interval):
      Similar to `adjust_time_for_diffusion` but specifically adjusted to support testing the universality hypothesis
      of network evolution under varying conditions of effective diffusion.

Usage:
    The functions within this module are typically used to initialize or modify the settings of a simulation before
    its execution. By importing this module, other parts of the project can easily apply consistent configurations,
    modify them or reset them based on the needs of specific experiments or analyses.

Example:
    # Importing the module
    from configuration import adjust_time_for_diffusion

    # Setting initial configuration
    simulation_parameters = {
        'dt': 0.001,
        'time_end': 200,
        'timeSteps': 10000,
        # additional parameters
    }

    # Adjusting parameters for a specific diffusion rate
    adjusted_params = adjust_time_for_diffusion(simulation_parameters, diffusion=5)

Note:
    This module should be maintained with clear documentation for each function and any changes to the
    parameters' handling should be tested to ensure they do not affect the integrity of the simulations.
"""


def set_network_evolution_parameters(sim_config, effective_diffusion, diffusion=5.0):
    """
    Configures network evolution parameters by calculating the beta value required to achieve a specified effective
    diffusion rate (Deff) given a diffusion rate (D). This function updates the simulation configuration dictionary
    with the diffusion rate, effective diffusion, and the calculated beta value.

    This setup is crucial for studies where understanding the interaction between diffusion and edge weight adjustment
    dynamics is essential, such as validating theoretical models or conducting sensitivity analyses.

    Args:
        sim_config (dict): A dictionary where network parameters are stored and updated.
        effective_diffusion (float): The target effective diffusion rate (Deff), which is a key parameter in the model
            influencing the network dynamics. This rate combines the effects of node state diffusion (D) and edge
            weight adjustments.
        diffusion (float, optional): The base diffusion rate (D) used in the simulation, defaulted to 5.0. This
            represents the rate at which states propagate through the network independently of edge adjustments.

    Returns:
        dict: The updated simulation configuration dictionary with the newly set parameters.

    Updates:
        - 'D': Sets the diffusion rate in the configuration.
        - 'Deff': Sets the effective diffusion rate.
        - 'beta': Sets the beta parameter calculated to achieve the specified effective diffusion given the diffusion
          rate. Beta reflects the rate at which edge weights are adjusted in response to state differences between
          connected nodes.

    Note:
        - It is critical to ensure that the effective_diffusion rate provided is attainable with the given diffusion
        parameter, as incorrect configurations can lead to unrealistic or unstable simulation behavior.
    """
    beta = diffusion / effective_diffusion

    sim_config.update({
        "D": diffusion,
        "Deff": effective_diffusion,
        "beta": beta,
    })

    return sim_config
def adjust_time_for_diffusion(sim_config,
                              diffusion,
                              base_dt=0.001,
                              base_time_end=200,
                              check_interval=200,
                              draw_interval=200,
                              update_interval=200):
    """
    Scales time-related simulation parameters based on the diffusion rate, ensuring time steps and durations
    are adjusted to reflect changes in state propagation speeds due to diffusion. This adjustment helps maintain
    consistent simulation dynamics.

    This function scales the simulation timings based on the diffusion rate (D), where D represents the rate of
    change propagation across the network. While this function uses the diffusion rate directly, it is important
    to understand the concept of the effective diffusion rate (Deff), which is D divided by β (beta), the rate of
    edge weight adjustments. Deff provides a conceptual understanding of the interaction between node state
    diffusion and edge dynamics.

    Args:
        sim_config (dict): Configuration dictionary for simulation parameters.
        diffusion (float): Diffusion rate D, used directly to scale time parameters.
        base_dt (float): Initial time step before scaling.
        base_time_end (int): Initial total simulation time before scaling.
        check_interval (int): How many times you want to stability phase checks.
        draw_interval (int):  How many times you want to redraw of the network graph.
        update_interval (int): How many times you want to update data during simulation.

    Returns:
        dict: Updated simulation configuration with adjusted time parameters.

    Notes:
        - Deff (D/β) is not directly used in this function but is crucial for understanding how diffusion and
          edge dynamics interact in the broader model. β (beta) is typically defined elsewhere in the model where
          edge weights are dynamically adjusted based on node interactions.
        - The file `initNetworkSame_checkHypothesis.py` shows teh universal graph evolution with same values of
          effective diffusion (Deff).
        - The file `initNetwork_dt.py` shows the maximal value of `dt`, which gives us stable evolution of the
          network. It seems a good choice is: `0.001 >= dt`.
    """
    scale_factor = 5 / diffusion
    adjusted_dt = base_dt * scale_factor
    adjusted_time_end = base_time_end * scale_factor
    total_steps = int(adjusted_time_end / adjusted_dt)

    sim_config.update({
        "dt": adjusted_dt,
        "time_end": adjusted_time_end,
        "timeSteps": total_steps,
        "timeStepsToCheck": int(total_steps / check_interval),
        "timeStepsDraw": int(total_steps / draw_interval),
        "timeStepsUpdateData": int(total_steps / update_interval)
    })

    return sim_config

def adjust_time_for_diffusion_vol2(sim_config,
                                   diffusion,
                                   base_dt=0.001,
                                   base_time_end=200,
                                   check_interval=200,
                                   draw_interval=200,
                                   update_interval=200):
    """
    Adjusts time-related simulation parameters based on the diffusion rate, focusing on testing the hypothesis
    that network evolution shows universal characteristics when scaled by the effective diffusion rate (Deff = D/β).
    This function modifies the simulation configuration to reflect changes in temporal dynamics influenced by
    diffusion, thereby facilitating the exploration of this hypothesis in network behavior.

    Args:
        sim_config (dict): Configuration dictionary for simulation parameters.
        diffusion (float): Diffusion rate D, directly used to scale time parameters, particularly the total simulation
            time.
        base_dt (float): Initial time step before scaling, remains constant in this function to isolate the effect of
            diffusion on total time.
        base_time_end (int): Initial total simulation time before scaling.
        check_interval (int): Base number of steps between stability phase checks.
        draw_interval (int): Base number of steps between network redraws.
        update_interval (int): Base number of steps between updates of simulation data.

    Returns:
        dict: Updated simulation configuration with adjusted time parameters, allowing for testing of the universality
            hypothesis in network evolution.

    Notes:
        - Deff (D/β) is conceptually crucial for understanding how diffusion and edge dynamics interact in the model.
          While β is not explicitly used in this function, it is inherently part of understanding Deff's role.
        - The function scales the number of steps for checks, drawing, and updates proportionally to how the total
          simulation time is adjusted,
          ensuring that the observational granularity adapts with the changes in the simulation timeline.
        - This function supports the research documented in `initNetworkSame_checkHypothesis.py`, where the evolution
          of the network is analyzed under different scenarios to validate the universality of Deff in determining
          network dynamics.
        - The file `initNetwork_dt.py` shows the maximal value of `dt`, which gives us stable evolution of the
          network. It seems a good choice is: `0.001 >= dt`.
    """
    # Calculate scale factor based on the diffusion rate
    scale_factor = 5 / diffusion

    # Adjusted time step remains constant; total simulation time is scaled
    adjusted_dt = base_dt
    adjusted_time_end = base_time_end * scale_factor
    total_steps = int(adjusted_time_end / adjusted_dt)

    # Adjusting intervals for various checks and updates based on scale factor
    sim_config.update({
        "dt": adjusted_dt,
        "time_end": adjusted_time_end,
        "timeSteps": total_steps,
        "timeStepsToCheck": total_steps // (check_interval * scale_factor),
        "timeStepsDraw": total_steps // (draw_interval * scale_factor),
        "timeStepsUpdateData": total_steps // (update_interval * scale_factor)
    })

    return sim_config
