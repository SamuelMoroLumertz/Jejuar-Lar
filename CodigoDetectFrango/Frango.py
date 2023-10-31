import time
class Frango:
    def __init__(self, xmin, ymin, xmax, ymax,id):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.id = id
        self.last_updated = time.time()

    def is_close(self, xmin, ymin, xmax, ymax):
        limit = 30
        if abs(self.xmin - xmin) < limit and abs(self.ymin - ymin) < limit and \
                abs(self.xmax - xmax) < limit and abs(self.ymax - ymax) < limit:
            return True
        return False

    def update_bounding_box(self, xmin, ymin, xmax, ymax):
        # Atualizar as coordenadas da bounding box do frango
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.last_updated = time.time()

    def is_inactive(self, inactive_threshold):
        current_time = time.time()
        elapsed_time = current_time - self.last_updated
        if elapsed_time > inactive_threshold:
            return True
        return False