def change_friction(cfg_dict, env_name):
    if env_name == 'friction0':
        cfg_dict['task']['sim']['default_physics_material']['static_friction'] = 0.5
        cfg_dict['task']['sim']['default_physics_material']['dynamic_friction'] = 2.0
        print(f'{env_name} config was changed successfully!')
    elif env_name == 'friction1':
        cfg_dict['task']['sim']['default_physics_material']['static_friction'] = 0.1
        cfg_dict['task']['sim']['default_physics_material']['dynamic_friction'] = 0.5
        print(f'{env_name} config was changed successfully!')

def initialize_task(config, env, init_sim=True):

    from tasks.robogame_task import RoboGameTask
    # from tasks.robogame_task2 import RoboGameTask2
    # from tasks.robogame_task3 import RoboGameTask3
    from tasks.robogame_task_search import RoboGameTaskSearch

    # Mappings from strings to environments
    if 'env_type' in config and config['env_type'] == 'search':
        task_map = {
            "RoboGame": RoboGameTaskSearch,
        }
        config['task']['env']['numAgents'] = 2
        config['num_agents'] = 2
    else:
        task_map = {
            "RoboGame": RoboGameTask,
        }

    change_friction(config, config['env_name'])

    from .config_utils.sim_config import SimConfig
    sim_config = SimConfig(config)

    cfg = sim_config.config
    task = task_map[cfg["task_name"]](
        name=cfg["task_name"], sim_config=sim_config, env=env
    )
    env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=init_sim)

    return task