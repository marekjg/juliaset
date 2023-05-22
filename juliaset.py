# Based on Taichi official tutorial but with dataclass

import taichi as ti
import taichi.math as tm


@ti.data_oriented
class JuliaSet:
    def __init__(self, n=640, result_dir='results', save_video=False):
        self.n = n
        self.pixels = ti.field(dtype=float, shape=(2 * n, n))
        self.gui = ti.GUI("Julia Set", res=(2 * n, n))
        self.save_video = save_video
        if save_video:
            self.video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)

    @ti.func
    def complex_sqr(self, z):
        return tm.vec2(z[0] * z[0] - z[1] * z[1], 2 * z[0] * z[1])

    @ti.kernel
    def paint(self, theta: float):
        for i, j in self.pixels:
            c = 0.7885 * tm.vec2(tm.cos(theta), tm.cos(theta))
            z = tm.vec2(i / self.n - 1, j / self.n - 0.5) * 2

            iterations = 0
            while z.norm() < 20 and iterations < 50:
                z = self.complex_sqr(z) + c
                iterations += 1
            self.pixels[i, j] = 1 - iterations * 0.02

    def simulate(self, dtheta=1e-2):
        theta = 0
        dtheta = dtheta
        while self.gui.running:
            self.paint(theta)
            self.gui.set_image(self.pixels)
            self.gui.show()
            if self.save_video:
                self.video_manager.write_frame(self.pixels.to_numpy())
            theta += dtheta
        
        if self.save_video:
            self.video_manager.make_video(gif=True, mp4=True)


if __name__ == "__main__":
    ti.init(arch=ti.cuda)
    juliaset = JuliaSet(save_video=True)
    juliaset.simulate()
