import numpy as np
import chumpy as ch
from chumpy import Ch
import scipy.sparse as sp
import matplotlib.pyplot as plt


class Snap(Ch):
    dterms = ('x',)

    def define_params(self):
        self.x0 = 2
        self.alpha = 8 # sigmoide slope
        self.beta = 2.5 # stich absice between the sigmoid and the quadratic curve
        self.sig = lambda x : 1/(1+ch.exp(-self.alpha*(x-self.x0)))

    def compute_r(self):
        self.define_params()
        values = np.zeros(self.x.shape)
        left_mask = self.x.r < (self.x0 + 0.5)
        right_mask = self.x.r >= (self.x0 + 0.5)
        values[left_mask] = self.sig(self.x.r[left_mask])
        values[right_mask] = ch.power((self.x.r[right_mask] - self.beta), 2) + self.sig(self.beta)
        return values

    def compute_dr_wrt(self, wrt):
        self.define_params()
        if wrt is self.x:
            values = np.zeros(self.x.shape)
            left_mask = self.x.r < (self.x0 + 0.5)
            right_mask = self.x.r >= (self.x0 + 0.5)
            if np.any(left_mask):
                values[left_mask] = np.diagonal(self.sig(self.x[left_mask]).dr_wrt(self.x[left_mask]).toarray())
            values[right_mask] = ch.power((self.x[right_mask] - self.beta), 2).dr_wrt(self.x).data + self.sig(self.beta)
            return sp.diags([values.ravel()], [0])


class Snap_4mm(Snap):
    def define_params(self):
        Snap.define_params(self)
        self.x0 = 4


class Snap_flat(Ch):
    dterms = ('x',)

    def define_params(self):
        self.x0 = 2
        self.alpha = 0.00000

    def left_fct(self, x):
        return self.alpha * x

    def right_fct(self, x):
        return ch.power(x - self.x0, 2) + self.left_fct(self.x0)

    def compute_r(self):
        self.define_params()
        values = np.zeros(self.x.shape)
        left_mask = self.x.r < (self.x0 )
        right_mask = self.x.r >= (self.x0)
        values[left_mask] = self.left_fct(self.x.r[left_mask])
        values[right_mask] = self.right_fct(self.x.r[right_mask])
        return values

    def compute_dr_wrt(self, wrt):
        self.define_params()
        if wrt is self.x:
            values = np.zeros(self.x.shape)
            left_mask = self.x.r < (self.x0)
            right_mask = self.x.r >= (self.x0)
            if np.any(left_mask):
                values[left_mask] = self.left_fct(self.x.r[left_mask])
            values[right_mask] = self.right_fct(self.x[right_mask]).dr_wrt(self.x).data
            return sp.diags([values.ravel()], [0])


class Snap_flat_4mm(Snap_flat):
    def define_params(self):
        Snap_flat.define_params(self)
        self.x0 = 4


class Collide(Ch):
    dterms = ('x',)

    def define_params(self):
        self.alpha_quadratic = 1
        self.alpha = -16 # sigmoide slope
        self.beta = -0.1 # stich absice between the sigmoid and the quadratic curve
        self.sig = lambda x : Ch(1)/(1+ch.exp(-self.alpha*(x+self.beta)))

    def compute_r(self):
        self.define_params()
        values = np.zeros(self.x.shape)
        left_mask = self.x.r < self.beta
        right_mask = self.x.r >= self.beta

        values[left_mask] = self.alpha_quadratic * ch.power((self.x.r[left_mask] - self.beta), 2) + self.sig(self.beta)
        values[right_mask] = self.sig(self.x.r[right_mask])
        return values

    def compute_dr_wrt(self, wrt):
        self.define_params()
        if wrt is self.x:
            values = np.zeros(self.x.shape)
            left_mask = self.x.r < self.beta
            right_mask = self.x.r >= self.beta

            values[left_mask] = self.alpha_quadratic * ch.power((self.x[left_mask] - self.beta), 2).dr_wrt(self.x[left_mask]).data + self.sig(self.beta)
            if np.any(right_mask):
                values[right_mask] = np.diagonal(self.sig(self.x[right_mask]).dr_wrt(self.x[right_mask]).toarray())
            return sp.diags([values.ravel()], [0])


def plot_snap_collide_cost(plate2bone):
    #visualize cost curve
    x_min = -5
    x_max = 10
    x_vector_ch = ch.array(np.linspace(x_min,x_max,200))
    y_snap_vector_ch = plate_snap_cost(x_vector_ch)
    y_collide_vector_ch = Collide(x_vector_ch) * ch.sum(plate2bone < 0)

    plt.plot(x_vector_ch, y_snap_vector_ch.r, 'b', label="snap cost")
    plt.plot(x_vector_ch, y_collide_vector_ch.r, 'r', label="collision cost")
    plt.scatter(plate2bone.r, plate_snap_cost(plate2bone).r + plate_collide_cost(plate2bone).r, c='green')
    plt.xlabel("plate2bone distance")
    plt.ylabel("plate2bone cost")
    plt.suptitle("Contribution of each plate vertex to the plate2bone cost")

    plt.draw()
    plt.pause(0.0001)
    plt.clf()


def sigmoid(x, a, b):
    return 1.0 / (1 + ch.exp(-a*(x-b)))


def plate_collide_cost(plate2bone_signed_dist):
    return Collide(plate2bone_signed_dist) * ch.sum(plate2bone_signed_dist < 0)


def plate_snap_cost(plate2bone_signed_dist, min_dist=2):
    if min_dist==4:
        return Snap_flat_4mm(plate2bone_signed_dist)
    else:
        return Snap_flat(plate2bone_signed_dist)

