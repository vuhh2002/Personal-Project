
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
