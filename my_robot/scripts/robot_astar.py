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

		if row < row_tot-1 and not grid[row+1][col].value>=90:
			self.neighbors.append(grid[row+1][col])

		if row > 0 and not grid[row-1][col].value>=90:
			self.neighbors.append(grid[row-1][col])

		if col < col_tot-1 and not grid[row][col+1].value>=90:
			self.neighbors.append(grid[row][col+1])

		if col > 0 and not grid[row][col-1].value>=90:
			self.neighbors.append(grid[row][col-1])


def h(p1, p2):
	x1, y1 = p1
	x2, y2 = p2
	return abs(x1 - x2) + abs(y1 - y2)

## we are going to work on this part to make it work on our code...

def reconstruct_path(came_from, current, draw):
	while current in came_from:
		current = came_from[current]
		current.make_path()
		draw()


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
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()

		current = open_set.get()[2]
		open_set_hash.remove(current)

		if current == end:
			reconstruct_path(came_from, end)
			end.make_end()
			return True

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
					neighbor.make_open()
		if current != start:
			current.make_closed()

	return False


def make_grid(rows, cols, occ_grid):
	grid = []
	for i in range(rows):
		grid.append([])
		for j in range(cols):
			spot = vertex(i, j, rows, cols, occ_grid[i][j])
			grid[i].append(spot)
	return grid

def read_image():
	img = cv2.imread("test_map.jpg", 0)
	#cv2.imshow('image', img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	return img


occ_grid = read_image()
print(np.unique(occ_grid))
rows = occ_grid.shape[0]
cols = occ_grid.shape[1]
grid = make_grid(rows, cols, occ_grid)
for i in range(rows):
	for j in range(cols):
		grid[i][j].update_neighbors(grid)



# def main(win, width):
# 	ROWS = 10
# 	grid = make_grid(ROWS, width)

# 	start = None
# 	end = None

# 	run = True
# 	while run:
# 		draw(win, grid, ROWS, width)
# 		for event in pygame.event.get():
# 			if event.type == pygame.QUIT:
# 				run = False

# 			if pygame.mouse.get_pressed()[0]: # LEFT
# 				pos = pygame.mouse.get_pos()
# 				row, col = get_clicked_pos(pos, ROWS, width)
# 				spot = grid[row][col]
# 				if not start and spot != end:
# 					start = spot
# 					start.make_start()

# 				elif not end and spot != start:
# 					end = spot
# 					end.make_end()

# 				elif spot != end and spot != start:
# 					spot.make_barrier()

# 			elif pygame.mouse.get_pressed()[2]: # RIGHT
# 				pos = pygame.mouse.get_pos()
# 				row, col = get_clicked_pos(pos, ROWS, width)
# 				spot = grid[row][col]
# 				spot.reset()
# 				if spot == start:
# 					start = None
# 				elif spot == end:
# 					end = None

# 			if event.type == pygame.KEYDOWN:
# 				if event.key == pygame.K_SPACE and start and end:
# 					for row in grid:
# 						for spot in row:
# 							spot.update_neighbors(grid)

# 					algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end)

# 				if event.key == pygame.K_c:
# 					start = None
# 					end = None
# 					grid = make_grid(ROWS, width)

# 	pygame.quit()

# main(WIN, WIDTH)