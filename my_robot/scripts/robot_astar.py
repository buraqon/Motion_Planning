# Please check the code where alot of these parts came from:
# An amazing youtube video about visualizating A* algorithm (source code in description)
# https://www.youtube.com/watch?v=JtiK0DOeI4A&t=1434s

import math
from queue import PriorityQueue
import cv2
import numpy as np

class vertex:
	def __init__(self, row, col, total_rows, total_cols, value):
		self.row = row
		self.col = col
		self.neighbors = []
		self.value = value
		self.total_rows = total_rows
		self.total_cols = total_cols

	def get_pos(self):
		return self.row, self.col

	def update_neighbors(self, grid):
		self.neighbors = []
		row = self.row
		col = self.col
		row_tot = self.total_rows
		col_tot = self.total_cols

		# Corners

		if row < row_tot-1 and col < col_tot-1 and not grid[row+1][col+1].value==1:
			self.neighbors.append(grid[row+1][col+1])

		if row > 0 and col > 0 and not grid[row-1][col-1].value==1:
			self.neighbors.append(grid[row-1][col-1])

		if row < row_tot-1 and col > 0  and not grid[row+1][col-1].value==1:
			self.neighbors.append(grid[row+1][col-1])

		if row > 0 and col < col_tot-1 and not grid[row-1][col+1].value==1:
			self.neighbors.append(grid[row-1][col+1])

		# Horizontal, Vertical

		if row < row_tot-1 and not grid[row+1][col].value==1:
			self.neighbors.append(grid[row+1][col])

		if row > 0 and not grid[row-1][col].value==1:
			self.neighbors.append(grid[row-1][col])

		if col < col_tot-1 and not grid[row][col+1].value==1:
			self.neighbors.append(grid[row][col+1])

		if col > 0 and not grid[row][col-1].value==1:
			self.neighbors.append(grid[row][col-1])

def read_image():
	img = cv2.imread("test_map2.jpg", 0)


	#cv2.imshow('image', img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	return img

def make_grid(rows, cols, occ_grid):
	grid = []
	for i in range(rows):
		grid.append([])
		for j in range(cols):
			if(occ_grid[i][j]>=90):
				spot = vertex(i, j, rows, cols, 1) # 1 for barrrier
			else:
				spot = vertex(i, j, rows, cols, 0) # 0 for free space
			grid[i].append(spot)
	return grid

def h(p1, p2):
	x1, y1 = p1
	x2, y2 = p2
	return abs(x1 - x2) + abs(y1 - y2)

## we are going to work on this part to make it work on our code...

def reconstruct_path(came_from, current):
	while current in came_from:
		current = came_from[current]
		current.value = 4 # 4 for path

def algorithm(grid, start, end):
	count = 0
	open_set = PriorityQueue()
	open_set.put((0, count, start))
	came_from = {}
	g_score = {spot: float("inf") for row in grid for spot in row}
	g_score[start] = 0
	f_score = {spot: float("inf") for row in grid for spot in row}
	f_score[start] = h(start.get_pos(), end.get_pos())

	open_set_hash = {start}


	while not open_set.empty():
		current = open_set.get()[2]
		open_set_hash.remove(current)

		if current == end:
			reconstruct_path(came_from, end)
			return True, came_from

		for neighbor in current.neighbors:
			temp_g_score = g_score[current] + 1

			if temp_g_score < g_score[neighbor]:
				came_from[neighbor] = current
				g_score[neighbor] = temp_g_score
				f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
				if neighbor not in open_set_hash:
					count += 1
					open_set.put((f_score[neighbor], count, neighbor))
					open_set_hash.add(neighbor)
					#neighbor.make_open()
					neighbor.value = 2 # 2 for open
		if current != start:
			current.value = 3 # 3 for closed

	return False, came_from

# TESTING

def Astar(occ_grid, x1,y1, x2, y2):
	#occ_grid = read_image() # For testing only

	rows = occ_grid.shape[0]
	cols = occ_grid.shape[1]
	grid = make_grid(rows, cols, occ_grid)
	for i in range(rows):
		for j in range(cols):
			grid[i][j].update_neighbors(grid)


	cond, came_from = algorithm(grid, grid[x1][y1], grid[x2][y2])


	for c in came_from:
		if(c.value ==4):
			occ_grid[c.row][c.col] = 240

	# cv2.imshow('occ', occ_grid)
	# cv2.waitKey(0)

	return occ_grid, came_from
