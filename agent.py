import numpy as np
import utils


class Agent:
    def __init__(self, actions, Ne=40, C=40, gamma=0.7, display_width=18, display_height=10):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne  # used in exploration function
        self.C = C
        self.gamma = gamma
        self.display_width = display_width
        self.display_height = display_height
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self, model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None

    def update_n(self, state, action):
        # TODO - MP11: Update the N-table.
        if state is not None and action is not None :
            (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y,
             adjoining_body_top, adjoining_body_bottom, adjoining_body_left,
             adjoining_body_right) = state


            self.N[(food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y,
             adjoining_body_top, adjoining_body_bottom, adjoining_body_left,
             adjoining_body_right,action)] += 1
    def update_q(self, s, a, r, s_prime):
        # TODO - MP11: Update the Q-table.
        if s is not None and a is not None :
            (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y,
             adjoining_body_top, adjoining_body_bottom, adjoining_body_left,
             adjoining_body_right) =s
            alpha = self.C/(self.C + self.N[(food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y,
             adjoining_body_top, adjoining_body_bottom, adjoining_body_left,
             adjoining_body_right,a)])

            old_q = self.Q[s][a]
            max_q = np.max(self.Q[s_prime])
            temp_diff = r + self.gamma * max_q  -old_q
            self.Q[s][a] =  old_q + alpha* temp_diff

    def exploration_function(self, q_value, n_value):
        """
        Epsilon-greedy exploration function.
        """
        if n_value < self.Ne:
            return 1  # Encourage exploration
        else:
            return q_value  # Exploit learned values

    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        s_prime = self.generate_state(environment)
        reward = -0.1  # Initialize reward

        if self._train and self.s is not None and self.a is not None:
            # Check if the snake has just eaten a food pellet
            if points > self.points:
                reward = 1 # Reward for eating a food pellet

            # Check if the snake has just died
            if dead:
                reward = -1  # Negative reward for dying

            self.update_n(self.s, self.a)
            self.update_q(self.s, self.a, reward, s_prime)

        action = self.select_action(s_prime, dead)

        if not dead:
            self.s = s_prime  # Update state for next iteration
            self.a = action  # Update action for next iteration
            self.points = points  # Update points

        else:
            self.reset()  # Reset for the next game if snake is dead

        return action

    def select_action(self, state, dead):
        action_order = [3, 2, 1, 0]
        if dead:
            return None
        if self._train:

            q_values_with_exploration = [self.exploration_function(self.Q[state][action], self.N[state][action]) for action in
                        self.actions]

            # Reorder the Q-values with exploration based on the action priority
            q_values_with_exploration = [q_values_with_exploration[i] for i in action_order]
            action = action_order[np.argmax(q_values_with_exploration)]
        else:
            q_values = [self.Q[state][action] for action in self.actions]
            q_values = [q_values[i] for i in action_order]
            action = action_order[np.argmax(q_values)]

        return action

    def generate_state(self, environment):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions
        '''
        # a = []
        # UP = exploration(self.Q[s_prime][utils.UP], self.N[s_prime][utils.UP])
        # np.append(a, [utils.UP, UP])
        # DOWN = exploration(self.Q[s_prime][utils.DOWN], self.N[s_prime][utils.DOWN])
        # np.append(a, [utils.DOWN, DOWN])
        # RIGHT = exploration(self.Q[s_prime][utils.RIGHT], self.N[s_prime][utils.RIGHT])
        # np.append(a, [utils.RIGHT, RIGHT])
        # LEFT = exploration(self.Q[s_prime][utils.LEFT], self.N[s_prime][utils.LEFT])
        # np.append(a, [utils.LEFT, LEFT])
        snake_head_x = environment[0]
        snake_head_y = environment[1]
        snake_body =environment[2]
        food_x = environment[3]
        food_y =environment[4]
        rock_x = environment[5]
        rock_y = environment[6]

        if snake_head_x  < food_x :
            food_dir_x =2
        elif snake_head_x > food_x :
            food_dir_x = 1
        else:
            food_dir_x = 0

        if snake_head_y < food_y :
            food_dir_y =2
        elif snake_head_y > food_y :
            food_dir_y = 1
        else:
            food_dir_y = 0


        if ( snake_head_x == 1   ) \
                or( snake_head_y == rock_y   and snake_head_x == rock_x +2  )\
                or( snake_head_y == rock_y   and snake_head_x == rock_x -1 and snake_head_x == 1) \
            or( snake_head_y == rock_y   and snake_head_x == self.display_width-2 and snake_head_x == rock_x +2):

            adjoining_wall_x = 1
        elif (snake_head_x == rock_x-1 and snake_head_y == rock_y  )or snake_head_x == self.display_width-2:
            adjoining_wall_x =2
        else:
            adjoining_wall_x = 0

        # top wall down rock
        if snake_head_y ==1 \
            or (rock_x == snake_head_x and snake_head_y  ==rock_y + 1) \
            or (rock_x+1 == snake_head_x and snake_head_y == rock_y + 1) \
            or (rock_x == snake_head_x and snake_head_y  ==rock_y + 1 and snake_head_y ==self.display_height-2) \
            or (rock_x+1 == snake_head_x and snake_head_y == rock_y + 1 and snake_head_y == 1) \
            or (rock_x == snake_head_x and snake_head_y ==1 and snake_head_y  == rock_y - 1 ) \
            or (rock_x+1 == snake_head_x and snake_head_y == 1 and snake_head_y == rock_y - 1):

                adjoining_wall_y = 1
        elif (snake_head_y  == rock_y - 1 and rock_x <=snake_head_x <=rock_x+1) \
                or snake_head_y == self.display_height-2:
            adjoining_wall_y =2
        else:
            adjoining_wall_y = 0

        adjoining_body_top =0
        adjoining_body_bottom=0
        adjoining_body_left=0
        adjoining_body_right=0

        for i in snake_body:
            if snake_head_y - 1 == i[1] and snake_head_x == i[0]:
                adjoining_body_top =1
            if snake_head_y + 1 == i[1]and snake_head_x == i[0]:
                adjoining_body_bottom =1
            if snake_head_x - 1 == i[0] and snake_head_y == i[1]:
                adjoining_body_left =1
            if snake_head_x + 1 == i[0] and snake_head_y == i[1]:
                adjoining_body_right =1




        # TODO - MP11: Implement this helper function that generates a state given an environment
        new_state = (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y,
                  adjoining_body_top, adjoining_body_bottom, adjoining_body_left,
                  adjoining_body_right)

        return new_state
