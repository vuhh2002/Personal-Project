import numpy as np
import matplotlib.pyplot as plt


class Plotter():
    def __init__(self, forward, name=None):
        self.forward = forward
        self.name = name

    def space(self, start, end, dense=100):
        temp = np.linspace(start, end, dense)
        X, Y = np.meshgrid(temp, temp)

        Z = np.zeros(X.shape)
        for i in range(dense):
            for j in range(dense):
                x = (X[(i,j)], Y[(i,j)])
                Z[(i,j)] = self.forward(x)

        return X, Y, Z

    def show(self, start, end, invert_axis=True):
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
        contour = ax2.contourf(X,Y,Z, levels=50, cmap='jet', alpha=0.7)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')

        plt.show()