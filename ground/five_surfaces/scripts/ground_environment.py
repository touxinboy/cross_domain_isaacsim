"""Map contacts on the five tiles to their original L17 friction coefficients."""
import numpy as np


class FiveSurfaceEnvironment:
    def __init__(self, config):
        self.config = config
        self.default_ground_mu = config['surfaces'][0]['dynamic_friction']

    def surface_at(self, x, y):
        for tile in self.config['surfaces']:
            if (tile['x_min_m'] <= x <= tile['x_max_m']
                    and tile['y_min_m'] <= y <= tile['y_max_m']):
                return tile
        return None

    def ground_mu(self, x, y, collider):
        for tile in self.config['surfaces']:
            path = '/World/Ground/'+tile['id']
            if collider == path or collider.startswith(path+'/'):
                return tile['dynamic_friction']
        tile = self.surface_at(x, y)
        return tile['dynamic_friction'] if tile else self.default_ground_mu

    def water_columns(self, x, y):
        x = np.asarray(x)
        return np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
