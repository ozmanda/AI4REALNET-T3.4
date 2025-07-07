from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from flatland.envs.rail_env import RailEnvActions
from IPython.display import HTML, display, clear_output
import ipywidgets as ipw
import PIL
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib
import numpy as np
from io import BytesIO
import time
from tqdm import tqdm

seed = 42

def create_rendering_area():
    rendering_area = ipw.Image()
    return rendering_area

def render_env_to_image(flatland_renderer, show_obs):
    flatland_renderer.render_env(show=False, show_rowcols=True, show_observations=show_obs)
    image = flatland_renderer.get_image()
    return image

def render_env(flatland_renderer, rendering_area : ipw.Image, show_obs):
    pil_image = PIL.Image.fromarray(render_env_to_image(flatland_renderer, show_obs))
    if rendering_area is None:
        clear_output(wait=False)
        display(pil_image)
        return

    # convert numpy to PIL to png-format bytes
    with BytesIO() as fOut:
        pil_image.save(fOut, format="png")
        byPng = fOut.getvalue()

    # set image value to png bytes which updates the image in the browser
    rendering_area.value=byPng

def process_frames(frames, frames_per_second=20):
    dpi = 72
    interval = 1000 / frames_per_second # ms

    plt.figure(figsize=(frames[0].shape[1]/dpi, frames[0].shape[0]/dpi), dpi=dpi)
    ax = plt.gca()
    plt.axis=('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plot = plt.imshow(frames[0])

    def init():
        pass

    def update(i):
        plot.set_data(frames[i])
        return plot,

    anim = FuncAnimation(fig=plt.gcf(),
                      func=update,
                      frames=len(frames),
                      init_func=init,
                      interval=interval,
                      repeat=False,
                      repeat_delay=20,
                      cache_frame_data=True
                      )

    plt.close(anim._fig)
    return anim


def run_simulation(env, enable_in_simulation_rendering=False, controller=None, show_obs=False, sleep=0.5):
    # reset env to ensure it is freshly initialized
    obs, info = env.reset(regenerate_rail=False, regenerate_schedule=False, random_seed=seed)

    # Enter env
    action_dict = dict()
    for agent in env.get_agent_handles():
        action_dict.update({agent: RailEnvActions.MOVE_FORWARD})
    _, _, _, info = env.step(action_dict)

    # print env infos about the trains after initialization.
    print(info)

    print("Agents in the environment have to solve the following tasks: \n")
    for agent_idx, agent in enumerate(env.agents):
        print(
              "The agent with index {} has the task to go from its initial position {}, facing in the direction {} to its target at {}.".format(
                  agent_idx, agent.initial_position, agent.direction, agent.target))

    # define image properties as well as how to render agents
    env_renderer = RenderTool(env, gl="PILSVG",
                                agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                                show_debug=False,
                                screen_height=800 / env.width * env.height,
                                screen_width=800)

    if enable_in_simulation_rendering:
        # create area to render on
        rendering_area = create_rendering_area()
        # render initial env state
        render_env(env_renderer, rendering_area, show_obs)
        # display in output
        display(rendering_area)
    else:
        # store frames and return when finished, e.g. for composing video
        offscreen_rendered_frames = []
        # render initial state
        offscreen_rendered_frames.append(render_env_to_image(env_renderer, show_obs))

    # the env builder sets a defalut max steps for an episode
    # dones["__all__"] is set to true if _max_episode_steps is reached
    for _ in tqdm(range(env._max_episode_steps - 1)):
        for a in range(env.get_num_agents()):
            action = controller.act(obs[a])
            action_dict.update({a: action})

        obs, rewards, dones, info = env.step(action_dict)

        if dones["__all__"]: # time is up or all agents in targets
          break

        # rendering on screen or save frame for later
        if enable_in_simulation_rendering:
            render_env(env_renderer, rendering_area, show_obs)
            # slow down to see trains moving
            time.sleep(sleep)
        else:
            offscreen_rendered_frames.append(render_env_to_image(env_renderer, show_obs))

    # show video of run
    if not enable_in_simulation_rendering:
        fps = 20
        anim = process_frames(offscreen_rendered_frames, frames_per_second=fps)
        writer = PillowWriter(fps=fps,
                                   metadata=dict(artist='Flatland'),
                                   bitrate=1800)
        anim.save('flatland-env.gif', writer=writer)
        return display(HTML(anim.to_jshtml()))