import os
import copy
import time
import queue
import random
from IPython.display import clear_output

objects = sorted(['W','S','C','Shp']) # sorted() asures that objects are in alphabetical order
island = sorted([])
mainland = copy.deepcopy(objects)

print(f"Initially on mainland there is {mainland} and on island there is {island}")

def shepherd_on_island(island):
    """
    On this function it is checked if shepherd is on island
    if shepherd is on island returns True otherwise False
    """
    if "Shp" in island:
        return True
    return False

def objects_on_island(objects, mainland):
    """
    On this function we check which objects are on island when objects on mainland are known.
    Deletes all objects from objects list that are on mainland
    """
    return [obj for obj in objects if obj not in mainland]

def permited_state(objects, mainland):
    """
    On this function it is checked if the planned move is allowed on both mainland and on island
    if move is not allowed function returns False and if move is allowed returns True
    """
    
    island = objects_on_island(objects, mainland)
    # if shepherd is on island, (wolf and sheep) or (sheep and cabbage) can't be on mainland
    if shepherd_on_island(island):
        if ("W" in mainland and "S" in mainland) or ("S" in mainland and "C" in mainland):
            return False
        return True
    # Using same rules as above but this time shepherd is on mainland
    else:
        if ("W" in island and "S" in island) or ("S" in island and "C" in island):
            return False
        return True
    
def generate_new_permited_mainlands(objects, mainland):
    "This function will generate new permited states for mainland"
    mainland = copy.deepcopy(mainland)
    new_mainlands = [] # initialize new_mainlands list
    island = objects_on_island(objects, mainland)   
    if shepherd_on_island(island):
        island.remove("Shp") # atleast shepherd is moving out of island
        mainland.append("Shp")
        if permited_state(objects, mainland):
            new_mainlands.append(sorted(mainland))
        for obj in island:
            mainland_tmp = mainland + [obj]
            if permited_state(objects, mainland_tmp):
                new_mainlands.append(sorted(mainland_tmp))
                
    else:
        mainland.remove("Shp") # Atleast shepherd leaves mainland
        island.append("Shp")
        if permited_state(objects, mainland):
            new_mainlands.append(sorted(mainland))
        mainland_tmp = copy.deepcopy(mainland)
        for obj in mainland:
            mainland_tmp.remove(obj)
            if permited_state(objects, mainland_tmp):
                new_mainlands.append(sorted(mainland_tmp))
            mainland_tmp.append(obj)
    return new_mainlands

def print_moves(objects, path):
    """
    This function prints complete path once path is found.
    """
    print('    Mainland' + ' '*23 + 'Boat' +' '*28 + 'Island')
    for i, mainland in enumerate(path):
        island = objects_on_island(objects, mainland)
        if shepherd_on_island(island):
            print('     {}'.format(mainland) + ' '*(61-len(str(mainland))) + '{}'.format(island))
            if i<len(path)-1:
                boat = [x for x in island if x not in objects_on_island(objects, path[i+1])]
                print('{}.   '.format(i+1) + ' '*27 + '<-{}'.format(boat))
        else:
            print('     {}'.format(mainland) + ' '*(61-len(str(mainland))) + '{}'.format(island))
            if i<len(path)-1:
                boat = [x for x in mainland if x not in path[i+1]]
                print('{}.   '.format(i+1) + ' '*29 + '{}->'.format(boat))

def depth_first(objects, mainland):
    """
    This function solves the river crossing puzzle using depth-first search
    """
    goal = []
    path = [mainland]
    #-------- WRITE HERE YOUR CODE --------
    open_list = []
    closed_list = []
    # 1. Initialize two empty lists: open_list and closed_list

    
    # 2. Define variable initial_state which is list that consists of two other lists mainland and path
    initial_state = [mainland,path]
    # 3. add initial_state to open_list
    open_list.append(initial_state)
    # 4. Loop through open_list until it is empty (Tip: while loop)
    while open_list:
        # 4.A choose last element of open_list
        mainland, path = open_list.pop()
        
        # 4.B if variable mainland is not in closed_list
        if mainland not in closed_list:
            
            # 4.B.a Add variable mainland to closed_list
            closed_list.append(mainland)
            
            # 4.B.b check if mainland is same as goal
            if mainland == goal:
                # 4.B.b.i if it is return path
                return path
            # 4.B.c Generatating variable new_mainlands using helper function generate_new_permited_mainlands
            new_mainlands = generate_new_permited_mainlands(objects, mainland)
        
            # 4.B.d Loop through new_mainlands one at a time using loop variable new_mainland
            for new_mainland in new_mainlands:
                # 4.B.d.i Solve variable new_path by adding to list variables path and new_mainland
                new_path = path + [new_mainland]
                # 4.B.d.ii Append to open_list new state which consists of variables new_mainland and new_path
                open_list.append([new_mainland, new_path])
            
    # 5. print error message when open_list is empty and no solution found
    print("Solution couldn't be found.")
    #-----------------------------------

def breadth_first(objects, mainland):
    """
    This function solves river crossing puzzle using breadth-first search.
    Solution is simalr to depth-first search only exception being on pseudocode's part 4.A you pick
    first element of the list instead of last element.
    """
    goal = []
    path = [mainland]
    #-------- WRITE HERE YOUR CODE --------
    # 1. Initialize two empty lists: open_list and closed_list
    open_list = []
    closed_list = []
    # 2. Define variable initial_state which is list that consists of two other lists mainland and path
    initial_state = [mainland,path]
    # 3. add initial_state to open_list
    open_list.append(initial_state)
    # 4. Loop through open_list until it is empty (Tip: while loop)
    while open_list:
        # 4.A choose first element of open_list (Tip: .pop(0) list method)
        mainland, path = open_list.pop(0)
        # 4.B if variable mainland is not in closed_list
        if mainland not in closed_list:
            # 4.B.a Add variable mainland to closed_list
            closed_list.append(mainland)
            # 4.B.b check if mainland is same as goal
            if mainland == goal:
                # 4.B.b.i if it is return path
                return path
            # 4.B.c Generatating variable new_mainlands using helper function generate_new_permited_mainlands
            new_mainlands = generate_new_permited_mainlands(objects, mainland)
        
            # 4.B.d Loop through new_mainlands one at a time using loop variable new_mainland
            for new_mainland in new_mainlands:
                # 4.B.d.i Solve variable new_path by adding to list variables path and new_mainland
                new_path = path + [new_mainland]
            
                # 4.B.d.ii Append to open_list new state which consists of variables new_mainland and new_path
                open_list.append([new_mainland, new_path])
            
    # 5. print error message when open_list is empty and no solution found
    print("Solution couldn't be found.")
    #-----------------------------------
    
depth_first_path = depth_first(objects, mainland)
print_moves(objects, depth_first_path)

breadth_first_path = breadth_first(objects, mainland)
print_moves(objects, breadth_first_path)
