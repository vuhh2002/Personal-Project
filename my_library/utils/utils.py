
def load_xy(monitor_dir, num_timesteps=None):
  from stable_baselines3.common.results_plotter import load_results

  df = load_results(monitor_dir)
  df.drop('t', inplace=True, axis=1)
  df.drop('index', inplace=True, axis=1)

  df['l'] = df.l.cumsum()
  if num_timesteps != None:
    df = df[df.l <= num_timesteps]
  
  x = df.l.values
  y = df.r.values
  return x, y


def estimate_y(x, prev_point, next_point):
  y = prev_point[1] + (prev_point[1] - next_point[1]) * (x - prev_point[0]) / (prev_point[0] - next_point[0])
  return y


def env_plot(env_id, monitor_dirs_of_algos, title=True, num_timesteps=1e6, alpha=0.5):
  import os
  import matplotlib.pyplot as plt

  
  for algo, monitor_dirs in monitor_dirs_of_algos.items():
    lines = []
    for monitor_dir in monitor_dirs:
      assert os.path.exists(monitor_dir), f'{monitor_dir} is not exist'
      assert len(os.listdir(monitor_dir)) != 0, f'{monitor_dir} have no *.monitor.csv files'
      x, y = load_xy(monitor_dir)
      assert x[-1] >= num_timesteps, f'trained timesteps: {x[-1]} < num_timesteps: {num_timesteps}'    
      line = (x, y)
      lines.append(line)

    
    list_x, mean_y, lower_y, upper_y = line_distribution(lines, num_timesteps)

    plt.plot(list_x, mean_y, label=algo)
    plt.fill_between(list_x, lower_y, upper_y, alpha=alpha)

  if title:
    plt.title(env_id)
  plt.xlabel('Timesteps')
  plt.ylabel('Average Reward')
  plt.legend()
  plt.show()


def line_distribution(lines, num_timesteps):

  n = len(lines)
  max_timestep = num_timesteps
  list_x = []
  mean_y = []
  lower_y = []
  upper_y = []

  min_timestep = max_timestep
  for line_index in range(n):
    line = lines[line_index]
    x = line[0]
    timestep0 = x[0]
    if min_timestep > timestep0:
      min_timestep = timestep0

  point_indexes = [0] * n
  timestep = min_timestep
  for line_index in range(n):
    line = lines[line_index]
    x = line[0]
    i = point_indexes[line_index]
    while x[i] < timestep:
      point_indexes[line_index] += 1
      i += 1

  while True:
    mu = 0
    for line_index in range(n):

      line = lines[line_index]
      x = line[0]
      y = line[1]
      i = point_indexes[line_index]

      if timestep < x[i]:
        yi = estimate_y(timestep, (x[i-1], y[i-1]), (x[i], y[i]))
      elif timestep == x[i]:
        yi = y[i]
      else:
        assert False, 'why timestep > x[i]?'

      mu += yi
    mu /= n
    
    sigma = 0
    for line_index in range(n):

      line = lines[line_index]
      x = line[0]
      y = line[1]
      i = point_indexes[line_index]

      if timestep < x[i]:
        yi = estimate_y(timestep, (x[i-1], y[i-1]), (x[i], y[i]))
      elif timestep == x[i]:
        yi = y[i]
      else:
        assert False, 'why timestep > x[i]?'

      sigma += abs(yi - mu)
    sigma /= n**0.5

    list_x.append(timestep)
    mean_y.append(mu)
    lower_y.append(mu - sigma)
    upper_y.append(mu + sigma)

    if timestep == max_timestep:
      break

    new_timestep = max_timestep
    for line_index in range(n):

      line = lines[line_index]
      x = line[0]
      i = point_indexes[line_index]
      
      if x[i] == timestep:
        assert len(x) != i+1, f'trained timesteps: {x[i]} < num_timesteps: {num_timesteps}'
        point_indexes[line_index] += 1
        i += 1
      elif x[i] < timestep:
        assert False, 'why x[i] < timestep'

      new_timestep = min(new_timestep, x[i])
    timestep = new_timestep

  return list_x, mean_y, lower_y, upper_y


def one_line_plot(monitor_dir, title):
    """
    plot the results
    :param monitor_dir: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    
    import matplotlib.pyplot as plt

    x, y = load_xy(monitor_dir)
    plt.plot(x, y)
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.title(title)
    plt.show()


def record_video(model, env_id, video_dir, play=True, deterministic=True):
  from colabgymrender.recorder import Recorder
  import gym
  
  env = gym.make(env_id)
  env = Recorder(env, video_dir)

  episode_rewards = 0
  done = False
  
  obs = env.reset()
  while not done:
      action, _states = model.predict(obs, deterministic=deterministic)
      obs, reward, done, info = env.step(action)
      episode_rewards += reward

  if play:
    env.play()


def show_gif(fname):
  from IPython import display
  import base64

  with open(fname, 'rb') as fd:
    b64 = base64.b64encode(fd.read()).decode('ascii')
  return display.HTML(f'<img src="data:image/gif;base64,{b64}" />')


def search_all_module_names(root_dir='my_library'):
    from collections import deque
    from os import scandir

    res = [root_dir]
    dirs = deque([root_dir])
    
    while len(dirs) > 0:
        dir = dirs.popleft()
        for f in scandir(dir):
            if f.name.startswith('__') or f.name.startswith('.'):
                continue
            if f.is_dir():
                dirs.append(f.path)

            split_by = '\\'
            if split_by not in f.path:
              split_by = '/'
            res.append('.'.join(f.path.split(split_by)))
            # Remove '.py'
            if not f.is_dir():
                res[-1] = res[-1][:-3]

    return res


def reload_all_modules(root_dir='my_library'):
  from importlib import reload, import_module
  from sys import modules

  for module_name in search_all_module_names(root_dir):
    import_module(module_name)
    reload(modules[module_name])


def normal_distribution(size, mu, cov):
  import numpy as np

  assert len(mu) == len(cov)
  epsilon = np.random.randn(size, len(mu))
  samples = np.array(mu) + epsilon * np.sqrt(cov)
  
  return samples


class CallCountWrapper():
  def __init__(self, function):
    self.function = function
    self.num_calls = 0
  def __call__(self, *args, **kwargs):
    self.num_calls += 1
    return self.function(*args, **kwargs)


def tuple_all_sublist(a_list, dtype=None):
  res = None
  try:
    a_list = list(a_list)
    res = tuple(tuple_all_sublist(sublist, dtype=dtype) for sublist in a_list)
  except:
    if dtype is None:
      res = a_list
    else:
      res = dtype(a_list)

  return res


def build_gif(img_paths, save_path, delete_imgs=False):
  import imageio
  import shutil

  with imageio.get_writer(save_path, mode='I') as writer:
    for img_path in img_paths:
        image = imageio.imread(img_path)
        writer.append_data(image)

  if delete_imgs:
    for img_path in img_paths:
      shutil.rmtree(img_path)


def make_gif_for_es(
  all_pops,
  space,
  save_path,
  opt_ind=None,
  ):
  import matplotlib.pyplot as plt
  import os
  import time
  import shutil

  tmp_path = 'tmp_' + str(time.time())
  os.makedirs(tmp_path)
  img_paths = [os.path.join(tmp_path, str(i) + '.png') for i in range(len(all_pops))]
  X, Y, Z = space

  for img_path, pop in zip(img_paths, all_pops):
      plt.figure(figsize=(6,6))
      plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
      plt.scatter(pop[:,0], pop[:,1], s=50, c='#FFB7C5')
      if opt_ind is not None:
        plt.scatter(opt_ind[0], opt_ind[1], s=50, c='#Ff0000')
      plt.xlabel('x')
      plt.ylabel('y')
      plt.savefig(img_path)
      plt.close()

  build_gif(img_paths=img_paths, save_path=save_path)
  shutil.rmtree(tmp_path)
