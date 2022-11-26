
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


def build_gif_from_numpy(imgs, gif_path, fps=24):
  import numpy as np
  from PIL import Image

  imgs = [Image.fromarray(img) for img in imgs]
  duration = 1000 // fps
  imgs[0].save(
    gif_path,
    save_all=True,
    append_images=imgs[1:],
    duration=duration,
    loop=0,
    )


def figure_to_numpy(fig):
  import numpy as np
  import matplotlib.pyplot as plt

  fig.canvas.draw()
  data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

  return data


def make_gif_for_es(
  all_pops,
  space,
  gif_path,
  opt_ind=None,
  minimum_generation=None,
  fps=6,
  ):
  import numpy as np
  import matplotlib.pyplot as plt

  if minimum_generation == None:
    minimum_generation = fps

  if opt_ind is not None:
    generation = len(all_pops) - 1
    while generation >= minimum_generation and np.allclose(all_pops[generation], opt_ind):
      generation -= 1
    all_pops = all_pops[:generation+1]

  generation = len(all_pops) - 2
  while generation >= minimum_generation and np.allclose(all_pops[generation], all_pops[generation + 1]):
    generation -= 1
  all_pops = all_pops[:generation+1]

  X, Y, Z = space
  imgs = []

  for pop in all_pops:
      fig = plt.figure(figsize=(6,6))
      plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
      plt.scatter(pop[:,0], pop[:,1], s=50, c='#FFB7C5')
      if opt_ind is not None:
        plt.scatter(opt_ind[0], opt_ind[1], s=50, c='#Ff0000')
      plt.xlabel('x')
      plt.ylabel('y')
      imgs.append(figure_to_numpy(fig=fig))
      plt.close()

  build_gif_from_numpy(imgs=imgs, gif_path=gif_path, fps=fps)

  
def set_random_seed(seed: int) -> None:
  """
  Seed the different random generators.
  :param seed:
  :param using_cuda:
  """
  
  import numpy as np
  import torch as th
  import random

  # Seed python RNG
  random.seed(seed)
  # Seed numpy RNG
  np.random.seed(seed)
  # seed the RNG for all devices (both CPU and CUDA)
  th.manual_seed(seed)


def mean_and_std(values):
  import numpy as np

  n = len(values)
  mean = np.mean(values)
  std = np.abs(np.array(values) - mean).sum() / np.sqrt(n)

  return mean, std
  

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


def line_distribution_v2(lines):
  # lines = [line1, line2, ...]
  # line = [x, y]
  # x is already sorted acendingly

  import numpy as np

  lines = np.array(lines)
  minimax_x = lines[:,0,-1].min()
  list_x, mean_y, lower_y, upper_y = line_distribution(lines=lines, num_timesteps=minimax_x)
  upper_y = np.array(upper_y)
  lower_y = np.array(lower_y)
  y_std = (upper_y - lower_y) / 2

  return list_x, mean_y, y_std


def estimate_y(x, prev_point, next_point):
  y = prev_point[1] + (prev_point[1] - next_point[1]) * (x - prev_point[0]) / (prev_point[0] - next_point[0])
  return y


def lines_plot(lines, labels, title, xlabel, ylabel, xticks=None, alpha=0.5, basex=None, basey=None):
  import matplotlib.pyplot as plt
  from matplotlib.ticker import ScalarFormatter
  import numpy as np

  assert len(lines) == len(labels)
  fig, ax = plt.subplots()

  for i in range(len(lines)):
      x, mean_y, y_std = np.array(lines[i])
      plt.plot(x, mean_y, label=labels[i])
      plt.fill_between(x, mean_y - y_std, mean_y + y_std, alpha=alpha)

  if basex is not None:
    plt.xscale('log', basex=basex)
    ax.xaxis.set_major_formatter(ScalarFormatter())
  if basey is not None:
    plt.yscale('log', basey=basey)
    ax.yaxis.set_major_formatter(ScalarFormatter())
  if xticks is not None:
    ax.set_xticks(xticks)
  plt.grid(True)
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.legend()
  plt.show()
