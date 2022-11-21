import numpy as np
import matplotlib.pyplot as plt


class Benchmark():
    def __init__(self, forward, name=None, lower_bound=float('-inf'), upper_bound=float('+inf'), global_minima=None, f_x_star=None):
        self.forward = forward
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.global_minima = global_minima
        self.f_x_star = f_x_star
        self.name = name

    def space(self, start=float('-inf'), end=float('+inf'), dense=100):
        if start == float('-inf'):
            start = self.lower_bound
        assert start != float('-inf')
        if end == float('+inf'):
            end = self.upper_bound
        assert end != float('+inf')

        temp = np.linspace(start, end, dense)
        X, Y = np.meshgrid(temp, temp)

        Z = np.zeros(X.shape)
        for i in range(dense):
            for j in range(dense):
                x = (X[(i,j)], Y[(i,j)])
                Z[(i,j)] = self.forward(x)

        return X, Y, Z

    def show(self, start=float('-inf'), end=float('+inf'), invert_axis=True, save_path=None):
        X, Y, Z = self.space(start, end)
        plt.figure(figsize=(12,6))
        ax1 = plt.subplot(1,2,1, projection='3d')
        if invert_axis:
            ax1.invert_xaxis()
        surf = ax1.plot_surface(X,Y,Z, cmap='jet', alpha=0.7)
        
        if self.name != None:
            ax1.set_title(self.name)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

        ax2 = plt.subplot(1,2,2)
        contour = ax2.contourf(X, Y, Z, levels=50, cmap='jet', alpha=0.7)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')

        if save_path != None:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def __call__(self, x):
        x = np.clip(x, self.lower_bound, self.upper_bound)
        return self.forward(x)