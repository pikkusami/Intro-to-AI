import os
import copy
import time
import queue
import random
from IPython.display import clear_output

def print_state(state):
    """
    Creates printable game table
    """
    
    printout = ""
    printout += "+---+---+---+\n"
    for i in range(3):
        for j in range(3):
            tile = state[i*3+j]
            printout += "| {} ".format(" " if tile == 0 else tile)
        printout += "|\n"
        printout += "+---+---+---+\n"
    return printout

def solve_permited_moves(location):
    """
    This function solves moves for empty tile
    """
    
    moves = [1, -1, 3, -3]
    permited_moves = []
    for move in moves:
        if 0 <= location + move < 9:
            if move == 1 and (location == 2 or location == 5 or location == 8):
                continue
            
            if move == -1 and (location == 0 or location == 3 or location == 6):
                continue
                
            permited_moves.append(move)
    return permited_moves

def generate_new_permited_states(state):
    empty_tile = state.index(0) # empty tile
    permited_moves = solve_permited_moves(empty_tile)
    new_states = []
    
    for move in permited_moves:
        new_state = copy.deepcopy(state)
        (new_state[empty_tile + move], new_state[empty_tile]) = (new_state[empty_tile], new_state[empty_tile + move])
        new_states.append(new_state)
    return new_states

def calculate_heuristic_value(state, goal, heuristic_function):
    """
    Calculates value for estimation function depending on chosen heuristic function
    """
    if heuristic_function == "hamming":
        return calculate_hamming_distance(state, goal)
    elif heuristic_function == "city-block":
        return calculate_city_block_distance(state, goal)
    elif heuristic_function == "euclidean":
        return calculate_euclidean_distance(state, goal)
    else:
        print("Options are 'hamming', 'city-block' or 'euclidean'")
        
def sort_open_list(open_list, f_scores):
    f_scores_tmp = []
    for state, path in open_list:
        f_scores_tmp.append(f_scores[string(state)])
    return [x for y, x in sorted(zip(f_scores_tmp, open_list))]

def string(list_of_strings):
    """Converts list of strings in to string"""
    return "".join(list(map(str, list_of_strings)))

def a_star(initial_state, goal, heuristic_function):
    open_list = []
    closed_list = []
    path = []
    g_scores = {}
    f_scores = {}
    g_scores[string(initial_state)] = 0
    f_scores[string(initial_state)] = g_scores[string(initial_state)] + calculate_heuristic_value(initial_state, goal, heuristic_function)
    state = [initial_state, path]
    open_list.append(state)
    
    while len(open_list) != 0:
        open_list = sort_open_list(open_list, f_scores)
        state, path = open_list.pop(0)
        if state == goal:
            new_path = path + [state]
            return new_path, closed_list
        closed_list.append(state)
        new_states = generate_new_permited_states(state)
        for new_state in new_states:
            initial_g_score_value = g_scores[string(state)] + 1
            if new_state in closed_list and initial_g_score_value >= g_scores[string(new_state)]:
                continue
            if new_state not in closed_list or initial_g_score_value < g_scores[string(new_state)]:
                new_path = path + [state]
                g_scores[string(new_state)] = initial_g_score_value
                f_scores[string(new_state)] = g_scores[string(new_state)] + calculate_heuristic_value(new_state, goal, heuristic_function)
                if new_state not in open_list:
                    open_list.append([new_state, new_path])
    
    print("Solution couldn't be found")
    
def solve_game(initial_state, goal, heuristic_function):
    path, closed_list = a_star(initial_state, goal, heuristic_function)
    for state in path:
        clear_output(wait=True)
        print("Initial state:\n"+ print_state(initial_state))
        print(f"\nSolution with A*-search algorithm using {heuristic_function} distance:\n\n" + print_state(state))
        time.sleep(1)
    print(f"Solution was found in {len(path)-1} moves. And to find those algorithm searched in total {len(closed_list)} states")

def calculate_euclidean_distance(state, goal):
    """
    This function calculates euclidean distance from state to goal
    """
    
    # Initialize variable euclidean_value with 0
    euclidean_score = 0
    # Loop through 1 to 8
    for tile_score in range(1, 9):
        # Determine for the state and the target state where in the matrix the value of the tile is (e.g. state_x refers to the tile's horizontal coordinate)
        (state_x, state_y) = (state.index(tile_score) // 3, state.index(tile_score) % 3)
        (goal_x, goal_y) = (goal.index(tile_score) // 3, goal.index(tile_score) % 3)
        # Calculate for each tile squared horizontal and vertical distance and add their sum's square root into variable euclidean_score
        euclidean_score += ((goal_x-state_x)**2 + (goal_y - state_y)**2)**0.5
        
    return euclidean_score

# Define initial state and goal according to figure 2
initial_state = [7,2,3,1,0,6,5,8,4]
goal = [1,2,3,4,5,6,7,8,0]

# Check that function returns correct value
print('Euclideam distance for initial state is: {}'.format(calculate_euclidean_distance(initial_state, goal)))

# Solve game using euclidean distance
# solve_game(initial_state, goal, 'euclidean')

def calculate_hamming_distance(state, goal):
    """
    This function calculates hamming distance for target state
    """
    # -------- YOUR CODE HERE -----------
    # Initialize variable hamming_score with 0
    hamming_score = 0
    # Loop through tile's values from 1 to 8
    for tile_value in range(1, 9):
        # Figure out for state and goal what is the index of a value in given state (e.g state_index is index of tile's value in state parameter). Tip: .index())
        state_index = state.index(tile_value)
        goal_index = goal.index(tile_value)
        # if tile value's index is not equal to goal  value's index
        if state_index != goal_index:
            # increment hamming_score by 1 
            hamming_score += 1
    # at the end return hamming_score
    return hamming_score
    #-------------------------------------

# Confirm that right value is returned
print('Hamming-distance for initial state is: {}'.format(calculate_hamming_distance(initial_state, goal)))

# Solve game with hamming distance heuristic function
# solve_game(initial_state, goal, 'hamming')

def calculate_city_block_distance(state, goal):
    """
    This function calculates city_block_distance from state to goal
    """
    #-------- YOUR CODE HERE --------
    # Initialize variable city_block_score with 0
    city_block_score = 0
    # Loop through 1 to 8
    for tile_value in range(1,9):
        # Determine for the state and the goal state where in the MATRIX the value of the tile is (e.g. state_x is tile's horizontal coordinate and state_y is tile's vertical coordinate)        
        state_x, state_y = state.index(tile_value) // 3, state.index(tile_value) % 3
        goal_x, goal_y = goal.index(tile_value) // 3, goal.index(tile_value) % 3
        # Calculate for each tile horizontal and vertical absolute distances and add them to variable city_block_score
        city_block_score += abs(goal_x - state_x) + abs(goal_y - state_y)
    # Finaly return city_block_score
    return city_block_score
    #-------------------------------------

# Confirm that function returns correct value
print('City-block distance for initial state is: {}'.format(calculate_city_block_distance(initial_state, goal)))

# Solve game using city-block distance
solve_game(initial_state, goal, 'city-block')
