import numpy as np
import random
from copy import deepcopy
np.set_printoptions(precision=3)

# Agent
# Snake ={'x':random.randrange(0, self.WIDTH), 'y':random.randrange(0, self.HEIGHT),
#                  'direction':random.randrange(0, 4), 'size':2, 'max_speed':1, 'max_life':1, 'attack':1,
#                         'dead': False, 'tail':[[x,y],[x,y],...]}
class Snake():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.direction = random.randrange(0, 4)
        self.tail = list()
        self.dead = False

        self.life = 100
        self.size = 1

        self.reward = 0
        self.apples_eaten = 0
        self.score = 0

    def params(self):
        pars = {'x': self.x, 'y': self.y, 'direction': self.direction,
                'size': self.size, 'max_speed': self.max_speed, 'max_life': self.max_life,
                'attack': self.attack}
        return pars

    def add_tail(self, n=1):
        for _ in range(n):
            if len(self.tail) == 0:
                _x = self.x
                _y = self.y
            else:
                _x = self.tail[-1][0]
                _y = self.tail[-1][1]
            self.tail.append([_x - 0.05, _y])

            # cutting tail if no life or some1 eat it

    def cut_tail(self, i):
        self.tail = self.tail[:int(i)]


class game():
    # In: Width and Height of map, Snake object
    def __init__(self, width, height):
        self.WIDTH = int(width)
        self.HEIGHT = int(height)
        self.map = np.zeros((width, height, 2))  # 0-me, 1-apple
        self.Life_reduce = -6  # 0.75
        self.tail_lenght = 2

        # Scores
        self.score_death = -2
        self.score_apple = 2
        self.score_health = -0.2
        self.score_move = -0.02

        self.num_apples = 1
        self.apple_pos = [{'x': 0, 'y': 0} for _ in range(self.num_apples)]
        self.apple_size = 1
        for i in range(self.num_apples):
            self.summon_apple(i)

        self.snake = Snake(random.randrange(0, width), random.randrange(0, height))

        # creating tails
        # and calc life
        self.snake.add_tail(self.tail_lenght)
        self.snake.life = 100
        self.snake.dead = False

        self.renew_map()

    # In: num action (0-2) for snake: left, straight, right
    # Out: environment, reward, snake is dead?
    def step(self, action):
        self.snake.direction = (self.snake.direction + (action - 1)) % 4 # left, straight, right

        # move snake

        if not self.snake.dead:
            self.snake.reward = 0 + self.score_move
            # if snake died return False
            if self.snake.direction == 0:  # right
                x_move = 1; y_move = 0
            elif self.snake.direction == 1:  # down
                x_move = 0; y_move = 1
            elif self.snake.direction == 2:  # left
                x_move = -1; y_move = 0
            elif self.snake.direction == 3:  # up
                x_move = 0; y_move = -1

            # move snake tails
            ltx = self.snake.x
            lty = self.snake.y
            for j in range(len(self.snake.tail)):
                _ltx = self.snake.tail[j][0]
                _lty = self.snake.tail[j][1]
                self.snake.tail[j][0] = ltx
                self.snake.tail[j][1] = lty
                ltx = _ltx
                lty = _lty

            # move head
            self.snake.x += x_move
            self.snake.y += y_move

        # detect collision with tail
        for k in range(len(self.snake.tail)):
            if (int(self.snake.x) == int(self.snake.tail[k][0])
                    and int(self.snake.y) == int(self.snake.tail[k][1])):
                self.snake.reward += (len(self.snake.tail) - 1 - k) * self.score_health
                self.snake.cut_tail(k)
                self.snake.life = 100
                #                     print('cutted', self.number)
                break

        # teleport snake from side to side
        if self.snake.x < 0:
            self.snake.x += self.WIDTH
        elif self.snake.x > self.WIDTH - 1:
            self.snake.x -= self.WIDTH
        elif self.snake.y < 0:
            self.snake.y += self.HEIGHT
        elif self.snake.y > self.HEIGHT - 1:
            self.snake.y -= self.HEIGHT

        # apple eater
        for j, a in enumerate(self.apple_pos):
            size = (self.apple_size + self.snake.size) / 2
            if (abs(self.snake.x - a["x"]) < size
                    and abs(self.snake.y - a["y"]) < size):
                self.snake.apples_eaten += 1
                self.snake.add_tail()
                self.snake.reward += self.score_apple
                self.summon_apple(j)
                self.snake.life = 100

        # minus life
        self.snake.life += self.Life_reduce  # 0.75
        # self.snake.reward -= self.Life_reduce
        # health score
        #             self.score_f_health = (self.life+100*self.max_life*(len(self.tail))) * score_health
        if self.snake.life <= 0:
            if len(self.snake.tail) > 0:
                self.snake.cut_tail(-1)
                self.snake.life = 100
            else:
                self.snake.dead = True
                self.snake.reward += self.score_death


        # More REWARDS !!!
        # dist_to_apples = rew_apple_dist(s, self.apple_pos)
        # self.snake.reward += dist_to_apples

        # Total score

        self.snake.score += self.snake.reward

        # ---- move all snakes -----

        self.renew_map()
        return self.rotate_map(), self.snake.reward, self.snake.dead

    # putting elements on map
    def renew_map(self):
        self.map = np.zeros((self.WIDTH, self.HEIGHT, 2))  # 0-me, 1-apple

        for a in self.apple_pos:
            self.map[a['x'], a['y'], 1] = 1

        self.map[round(self.snake.x), round(self.snake.y), 0] = 1
        for s1 in self.snake.tail:
            self.map[round(s1[0]), round(s1[1]), 0] = 1

    def rotate_map(self):
        view = np.zeros((self.map.shape))
        # rotate map
        if self.snake.direction == 0:  # right direction
            # ranges width and height (rotation)
            w_r_mapp = min(view.shape[0], self.map.shape[1])
            h_r_mapp = min(view.shape[1], self.map.shape[0])

            # limits for map (rotation)
            w_mapp_min = self.snake.x - int(h_r_mapp / 2)
            w_mapp_max = w_mapp_min + h_r_mapp
            h_mapp_min = self.snake.y - int(w_r_mapp / 2)
            h_mapp_max = h_mapp_min + w_r_mapp

            # array for loop for map (rotation)
            w_mapp = np.arange(w_mapp_max, w_mapp_min, -1) % self.map.shape[0]
            h_mapp = np.arange(h_mapp_min, h_mapp_max, 1) % self.map.shape[1]

            # limits for view
            w_view_min = int(view.shape[0] / 2) - int(w_r_mapp / 2)
            w_view_max = w_view_min + w_r_mapp
            h_view_min = int(view.shape[1] / 2) - int(h_r_mapp / 2)
            h_view_max = h_view_min + h_r_mapp

            # array for loop for view
            w_view = np.arange(w_view_min, w_view_max, 1)
            h_view = np.arange(h_view_min, h_view_max, 1)

            for w in range(w_r_mapp):
                for h in range(h_r_mapp):
                    view[w_view[w], h_view[h]] = self.map[w_mapp[h], h_mapp[w]].copy()

        elif self.snake.direction == 2:  # left direction
            # ranges width and height (rotation)
            w_r_mapp = min(view.shape[0], self.map.shape[1])
            h_r_mapp = min(view.shape[1], self.map.shape[0])

            # limits for map (rotation)
            w_mapp_min = self.snake.x - int(h_r_mapp / 2)
            w_mapp_max = w_mapp_min + h_r_mapp
            h_mapp_min = self.snake.y - int(w_r_mapp / 2)
            h_mapp_max = h_mapp_min + w_r_mapp

            # array for loop for map (rotation)
            w_mapp = np.arange(w_mapp_min, w_mapp_max, 1) % self.map.shape[0]
            h_mapp = np.arange(h_mapp_max - 1, h_mapp_min - 1, -1) % self.map.shape[1]

            # limits for view
            w_view_min = int(view.shape[0] / 2) - int(w_r_mapp / 2)
            w_view_max = w_view_min + w_r_mapp
            h_view_min = int(view.shape[1] / 2) - int(h_r_mapp / 2)
            h_view_max = h_view_min + h_r_mapp

            # array for loop for view
            w_view = np.arange(w_view_min, w_view_max, 1)
            h_view = np.arange(h_view_min, h_view_max, 1)

            for w in range(w_r_mapp):
                for h in range(h_r_mapp):
                    view[w_view[w], h_view[h]] = self.map[w_mapp[h], h_mapp[w]].copy()

        elif self.snake.direction == 1:  # down direction
            # ranges width and height (rotation)
            w_r_mapp = min(view.shape[0], self.map.shape[0])
            h_r_mapp = min(view.shape[1], self.map.shape[1])

            # limits for map (rotation)
            w_mapp_min = self.snake.x - int(w_r_mapp / 2)
            w_mapp_max = w_mapp_min + w_r_mapp
            h_mapp_min = self.snake.y - int(h_r_mapp / 2)
            h_mapp_max = h_mapp_min + h_r_mapp

            # array for loop for map (rotation)
            w_mapp = np.arange(w_mapp_min, w_mapp_max, 1) % self.map.shape[0]
            h_mapp = np.arange(h_mapp_max - 1, h_mapp_min - 1, -1) % self.map.shape[1]

            # limits for view
            w_view_min = int(view.shape[0] / 2) - int(w_r_mapp / 2)
            w_view_max = w_view_min + w_r_mapp
            h_view_min = int(view.shape[1] / 2) - int(h_r_mapp / 2)
            h_view_max = h_view_min + h_r_mapp

            # array for loop for view
            w_view = np.arange(w_view_min, w_view_max, 1)
            h_view = np.arange(h_view_min, h_view_max, 1)

            for w in range(w_r_mapp):
                for h in range(h_r_mapp):
                    view[w_view[w], h_view[h]] = self.map[w_mapp[w], h_mapp[h]].copy()

        else:  # up direction
            # ranges width and height (rotation)
            w_r_mapp = min(view.shape[0], self.map.shape[0])
            h_r_mapp = min(view.shape[1], self.map.shape[1])

            # limits for map (rotation)
            w_mapp_min = self.snake.x - int(w_r_mapp / 2)
            w_mapp_max = w_mapp_min + w_r_mapp
            h_mapp_min = self.snake.y - int(h_r_mapp / 2)
            h_mapp_max = h_mapp_min + h_r_mapp

            # array for loop for map (rotation)
            w_mapp = np.arange(w_mapp_min, w_mapp_max, 1) % self.map.shape[0]
            h_mapp = np.arange(h_mapp_min, h_mapp_max, 1) % self.map.shape[1]

            # limits for view
            w_view_min = int(view.shape[0] / 2) - int(w_r_mapp / 2)
            w_view_max = w_view_min + w_r_mapp
            h_view_min = int(view.shape[1] / 2) - int(h_r_mapp / 2)
            h_view_max = h_view_min + h_r_mapp

            # array for loop for view
            w_view = np.arange(w_view_min, w_view_max, 1)
            h_view = np.arange(h_view_min, h_view_max, 1)

            for w in range(w_r_mapp):
                for h in range(h_r_mapp):
                    view[w_view[w], h_view[h]] = self.map[w_mapp[w], h_mapp[h]].copy()

        return view




    def summon_apple(self, ind):
        self.apple_pos[ind]['x'] = random.randrange(0, self.WIDTH)
        self.apple_pos[ind]['y'] = random.randrange(0, self.HEIGHT)

    def reset(self):

        self.snake.x = random.randrange(0, self.WIDTH)
        self.snake.y = random.randrange(0, self.HEIGHT)
        self.snake.cut_tail(0)
        self.snake.add_tail(self.tail_lenght)
        self.snake.score = 0
        self.snake.life = 100
        self.snake.dead = False
        self.apple_pos = [{'x':random.randrange(0, self.WIDTH), 'y':random.randrange(0, self.HEIGHT)} for _ in range(self.num_apples)]

        self.renew_map()
        return self.map


# reward for closing distanse to apples
def rew_apple_dist(snake, apples):
    rew = 0
    my_x = snake.x
    my_y = snake.y
    for a in apples:
        dif_xy = abs(a['x'] - my_x) + abs(a['y'] - my_y)
        rew += 3/(dif_xy+10) - 0.13

    return rew