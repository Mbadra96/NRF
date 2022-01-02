import pygame
from math import sin, cos, pi
from pathlib import Path  # type: ignore

from neuron.core.coder import SFDecoder, StepEncoder
from neuron.core.params_loader import TIME_STEP, SAMPLES, t
from neuron.optimizer.neat.genome import Genome
from neuron.simulation.inverted_pendulum import InvertedPendulum
from testing.model_ref_design import reference_model, reference_model_dot


class InvertedPendulumSimulation:
    def __init__(self):
        self.x_res = 720
        self.y_res = 500
        self.zoom = 10
        pygame.init()
        pygame.display.set_caption("Inverted Pendulum")
        self.screen = pygame.display.set_mode([self.x_res, self.y_res])  # TODO: ADD RESIZEABLE
        self.running = True

        # Constants
        self.cart_width = 2
        self.cart_height = 0.5
        self.ball_radius = 0.3
        self.rod_length = 2

        self.x = 0
        self.angle = 0

    def scale_x(self, x) -> int:
        return x * self.x_res / self.zoom

    def scale_y(self, y) -> int:
        return y * self.y_res / self.zoom

    def transform_x(self, x) -> int:
        des_x = self.x_res / 2
        des_x += x * self.x_res / self.zoom
        return int(des_x)

    def transform_y(self, y):
        # translate
        des_y = self.y_res / 2
        des_y -= y * self.y_res / self.zoom
        return int(des_y)

    def get_cords(self, x, y) -> (int, int):
        return self.transform_x(x), self.transform_y(y)

    def draw_line_from_angle(self):
        pygame.draw.line(self.screen, (0, 0, 0), (250, 250), (250, 400), 3)

    def draw_circle_from_angle(self):
        x = self.x + self.rod_length * sin(self.angle)
        y = cos(self.angle)
        radius = self.ball_radius * self.x_res / self.zoom

    def draw(self):

        x_cart, y_cart = self.get_cords(self.x - self.cart_width/2, 0)
        width = self.scale_x(self.cart_width)
        height = self.scale_y(self.cart_height)

        radius = self.ball_radius * self.x_res / self.zoom
        x_circle, y_circle = self.get_cords(self.x + self.rod_length * sin(self.angle), - self.rod_length * cos(self.angle))

        pygame.draw.line(self.screen, (0, 0, 0), (x_cart + self.scale_x(self.cart_width/2), y_cart + self.scale_y(self.cart_height/2)), (x_circle, y_circle), 3)
        pygame.draw.rect(self.screen, (0, 255, 0), (x_cart, y_cart, width, height))

        pygame.draw.circle(self.screen, (0, 0, 255), (x_circle, y_circle), radius)

    def run(self):
        encoder = StepEncoder()
        genome: Genome = Genome.load(f"{Path().absolute()}/scenarios/scenario_19/scenario_19")
        cont = genome.build_phenotype(TIME_STEP)


        theta_ref = pi
        theta_dot_ref = 0
        theta = pi - 0.1
        theta_dot = 0
        pen = InvertedPendulum(theta_0=theta)
        x = 0
        f_filt = 0
        for i in range(SAMPLES):
            pygame.time.delay(int(TIME_STEP*1000))
            e = (theta_ref - theta) + (theta_dot_ref - theta_dot)

            sensors = encoder.encode(e)
            action = cont.step(sensors, t[i], TIME_STEP)
            f = - 100 * action[0] + 100 * action[1]
            f_filt = f_filt + 0.1 * (f - f_filt)  # LOW PASS Filter
            theta, theta_dot, x, _ = pen.step(f_filt, t[i], TIME_STEP)  # Model
            self.angle = theta
            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    exit()
            print(x)
            self.screen.fill((255, 255, 255))
            self.draw()
            pygame.display.flip()

    def stop(self):
        pygame.quit()


if __name__ == '__main__':
    sim = InvertedPendulumSimulation()

    sim.run()

    sim.stop()

