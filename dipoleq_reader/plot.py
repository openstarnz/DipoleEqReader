
from matplotlib import pyplot as plt
class DPEqPlotter:
    def __init__(self):
        pass
    def plot_lcfs(self, ax=plt.gca(), **kwargs):
        coords = self.get_lcfs()
        ax.plot(coords[:, 0], coords[:, 1], **kwargs)
    def plot_fcfs(self, ax=plt.gca(), **kwargs):
        coords = self.get_fcfs()
        ax.plot(coords[:, 0], coords[:, 1], **kwargs)
    def plot_inner_wall(self, ax=plt.gca(), **kwargs):
        coords = self.get_inner_wall()
        ax.plot(coords[:, 0], coords[:, 1], **kwargs)
    def plot_outer_wall(self, ax=plt.gca(), **kwargs):
        coords = self.get_outer_wall()
        ax.plot(coords[:, 0], coords[:, 1], **kwargs)
    # def plot_eq(self, ax=plt.gca(), **kwargs):
    #     coords = self.get_eq()
    #     ax.plot(coords[:, 0], coords[:, 1], **kwargs)
