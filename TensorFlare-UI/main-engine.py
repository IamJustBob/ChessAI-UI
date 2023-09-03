import sys
import io
# original_stdout = sys.stdout
# sys.stdout = io.StringIO()
import pygame
import chess
from keras.models import load_model
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import random
import math

pygame.mixer.init()

move_sound = pygame.mixer.Sound('./sounds/move.wav')
cap_sound = pygame.mixer.Sound('./sounds/piece_taken.wav')
mate = pygame.mixer.Sound('./sounds/game_done.wav')

white = (240, 217, 181)
black = (181, 136, 99)
pygame.init()
screen = pygame.display.set_mode((720, 720))
pygame.display.set_caption("TensorFlare - Chess")
square_size = 90
decay_factor = 0.995
screen_width = 720
screen_height = 720

circle_color = (255, 255, 255)
circle_radius = 50
angle = 0 

# Load images
white_king_img = pygame.image.load('./chess_set/white/wK.png')
black_king_img = pygame.image.load('./chess_set/black/bK.png')

LIGHT_SQUARE_COLOR = white
DARK_SQUARE_COLOR = black
DARKEN_OVERLAY_COLOR = (0, 0, 0, 128)

def animate_loading_spinner():
	global angle

	# Clear the screen
	screen.fill((0, 0, 0))

	# Calculate circle's new position based on angle
	x = screen_width/2 + circle_radius * math.cos(angle)
	y = screen_height/2 + circle_radius * math.sin(angle)

	# Draw the circle
	pygame.draw.circle(screen, circle_color, (int(x), int(y)), 10)

	# Increment the angle
	angle += 0.05

	pygame.display.flip()

	# Small delay to control the speed of the animation
	pygame.time.wait(16)  # Roughly 60FPS

def display_end_menu(result, last_board_surface, turn):
	running = True
	button_font = pygame.font.Font(None, 48)

	# Define button dimensions and positions
	play_again_button = pygame.Rect(720 // 2 - 100, 720 // 2, 200, 50)
	quit_button = pygame.Rect(720 // 2 - 45, 720 // 2 + 60, 90, 50)

	while running:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
				pygame.quit()
				return
			if event.type == pygame.MOUSEBUTTONDOWN:
				if play_again_button.collidepoint(event.pos):
					# TODO: Restart the game
					return "again"
				elif quit_button.collidepoint(event.pos):
					running = False
					pygame.quit()
					return "quit"

		# Display the last board position
		screen.blit(last_board_surface, (0, 0))

		# Darken the screen
		darken_surface = pygame.Surface((720, 720), pygame.SRCALPHA)
		darken_surface.fill((0, 0, 0, 128))  # Semi-transparent black
		screen.blit(darken_surface, (0, 0))

		# Display Result
		font = pygame.font.Font(None, 74)
		if result == '1-0':
			if turn == "White":
				text = font.render('You Win!', True, (30, 255, 30))
			else:
				text = font.render('You Lose!', True, (255, 30, 30))
		elif result == '0-1':
			if turn == "White":
				text = font.render('You Lose!', True, (255, 30, 30))
			else:
				text = font.render('You Win!', True, (30, 255, 30))
		elif result == '1/2-1/2':
			text = font.render("It's a Draw!", True, (244, 244, 244))
		text_rect = text.get_rect(center=(720 // 2, 720 // 2 - 50))
		screen.blit(text, text_rect)

		# Display "Play Again" button
		pygame.draw.rect(screen, (100, 150, 100), play_again_button)
		play_text = button_font.render("Play Again", True, (255, 255, 255))
		play_text_rect = play_text.get_rect(center=play_again_button.center)
		screen.blit(play_text, play_text_rect)

		# Display "Quit" button
		pygame.draw.rect(screen, (150, 50, 50), quit_button)
		quit_text = button_font.render("Quit", True, (255, 255, 255))
		quit_text_rect = quit_text.get_rect(center=quit_button.center)
		screen.blit(quit_text, quit_text_rect)

		pygame.display.flip()

def draw_chessboard():
	square_size = 720 // 8
	for row in range(8):
		for col in range(8):
			color = LIGHT_SQUARE_COLOR if (row + col) % 2 == 0 else DARK_SQUARE_COLOR
			pygame.draw.rect(screen, color, pygame.Rect(col * square_size, row * square_size, square_size, square_size))

def darken_screen():
	overlay = pygame.Surface((720, 720), pygame.SRCALPHA)
	overlay.fill(DARKEN_OVERLAY_COLOR)
	screen.blit(overlay, (0, 0))

def display_color_choice(screen):
	draw_chessboard()  # Draw the normal chessboard
	darken_screen()    # Fill with white, or any other background color
	
	# Position the images/buttons in the center
	screen.blit(white_king_img, (160, 310))
	screen.blit(black_king_img, (460, 310))
	
	pygame.display.flip()

	running = True
	while running:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				exit()
			if event.type == pygame.MOUSEBUTTONDOWN:
				mouseX, mouseY = pygame.mouse.get_pos()
				if 160 <= mouseX <= 360 and 310 <= mouseY <= 410:  # Adjust these based on your image sizes
					return "White"
				elif 460 <= mouseX <= 660 and 310 <= mouseY <= 410:
					return "Black"

def display_promotion_options(screen, turn):
	# Translucent overlay to darken the board
	overlay = pygame.Surface((720, 720), pygame.SRCALPHA)
	overlay.fill((0, 0, 0, 128))
	screen.blit(overlay, (0, 0))

	# Load the images for each piece based on the turn
	white_dir = "./chess_set/white/"
	black_dir = "./chess_set/black/"
	if turn == chess.WHITE:
		queen_img = pygame.image.load(white_dir+'wQ.png')
		rook_img = pygame.image.load(white_dir+'wR.png')
		bishop_img = pygame.image.load(white_dir+'wB.png')
		knight_img = pygame.image.load(white_dir+'wN.png')
	else:
		queen_img = pygame.image.load(black_dir+'bQ.png')
		rook_img = pygame.image.load(black_dir+'bR.png')
		bishop_img = pygame.image.load(black_dir+'bB.png')
		knight_img = pygame.image.load(black_dir+'bN.png')
	
	# Determine the start position for the pieces (centered)
	start_x = (720 - 440) // 2
	start_y = (720 - 100) // 2

	# Display images
	screen.blit(queen_img, (start_x, start_y))
	screen.blit(rook_img, (start_x + 120, start_y))
	screen.blit(bishop_img, (start_x + 240, start_y))
	screen.blit(knight_img, (start_x + 360, start_y))
	
	pygame.display.flip()
	
	running = True
	while running:
		for event in pygame.event.get():
			if event.type == pygame.MOUSEBUTTONDOWN:
				mouseX, mouseY = pygame.mouse.get_pos()
				if start_x <= mouseX <= start_x + 100 and start_y <= mouseY <= start_y + 100:
					return 'q'
				elif start_x + 120 <= mouseX <= start_x + 220 and start_y <= mouseY <= start_y + 100:
					return 'r'
				elif start_x + 240 <= mouseX <= start_x + 340 and start_y <= mouseY <= start_y + 100:
					return 'b'
				elif start_x + 360 <= mouseX <= start_x + 460 and start_y <= mouseY <= start_y + 100:
					return 'n'



def get_reward(board):
	if board.is_game_over():
		if board.result() == '1-0':
			return 1
		elif board.result() == '0-1':
			return -1
		elif board.result() == "1/2-1/2":
			return -0.2
		else:
			return 0
	else:
		return 0

def mcts(board, model, epsilon, simulations):
	if random.uniform(0, 1) < epsilon:
		legal_moves = list(board.legal_moves)
		random_move = random.choice(legal_moves)
		return str(random_move) 
	# Initialize the root node
	root = Node(board.fen(), None)

	# Run the simulations
	for _ in range(simulations):
		# Select the leaf node
		leaf = root.select()

		# Expand the leaf node
		leaf.expand()

		# Simulate the game from the leaf node
		original_stdout = sys.stdout
		sys.stdout = io.StringIO()
		reward = simulate(leaf.board, model, verbose=False)

		# Backpropagate the reward
		leaf.backpropagate(reward)

	# Return the best move
	sys.stdout = original_stdout
	return root.best_move()

# Define the Node class for MCTS
class Node:
	def __init__(self, fen, parent):
		self.fen = fen
		self.parent = parent
		self.children = []
		self.visits = 0
		self.value = 0
	
	@property
	def board(self):
		return chess.Board(self.fen)
	
	def select(self):
		if not self.children:
			return self
		
		ucb1_values = [child.ucb1() for child in self.children]
		max_index = np.argmax(ucb1_values)
		return self.children[max_index].select()
	
	def expand(self):
		for move in self.board.legal_moves:
			new_board = self.board.copy()
			new_board.push(move)
			new_node = Node(new_board.fen(), self)
			self.children.append(new_node)
	
	def backpropagate(self, reward):
		self.visits += 1
		self.value += reward
		
		if self.parent:
			self.parent.backpropagate(reward)
	
	def ucb1(self):
		if not self.visits:
			return float('inf')
		
		exploitation = self.value / self.visits
		exploration = np.sqrt(2 * np.log(self.parent.visits) / self.visits)
		
		return exploitation + exploration
	
	def best_move(self):
		visits = [child.visits for child in self.children]
		max_index = np.argmax(visits)
		
		move_uci = list(self.board.legal_moves)[max_index].uci()
		
		return move_uci

# Define the simulate function for MCTS
def simulate(board, model, verbose=False):
	while not board.is_game_over():
		# Get the legal moves and their corresponding boards
		legal_moves = list(board.legal_moves)

		boards = []

		for move in legal_moves:
			new_board = board.copy()
			new_board.push(move)
			boards.append(new_board)

		# Predict the values of the boards using the DQN model
		values = model.predict(np.array([board_to_array(board) for board in boards]))

		# Select the best move based on the predicted values
		best_index = np.argmax(values)

		# Make the best move on the board
		board.push(legal_moves[best_index])

		if verbose:
			print(f"Move made: {legal_moves[best_index]} with value: {values[best_index]}")  # Hypothetical verbose output

	# Return the reward of the final board state
	return get_reward(board)

# Define a function to convert a chess board to a numpy array
def board_to_array(board):
	pieces_dict = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
				   'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6}
	
	board_array = np.zeros((1, 8, 8))
	
	for square in chess.SQUARES:
		piece = board.piece_at(square)
		
		if piece:
			row, col = divmod(square, 8)
			board_array[0, row, col] = pieces_dict[piece.symbol()]
	
	return board_array

def generate_chess_squares():
	squares = []
	for row in 'abcdefgh':
		for col in range(1, 9):
			squares.append(f'{row}{col}')
	return squares
from threading import Thread

def retrain():
	with open("engines/epsilon.txt","r",encoding='utf-8') as ep:
		epsilon = ep.read()
		epsilon = float(epsilon)
		decay_factor = 0.95
	sys.stdout = io.StringIO()
	history = list(game.move_stack)
	epsilon *= decay_factor
	for i in range(len(history)):
		# Get the current and next board states
		current_board = chess.Board()
		
		for move in history[:i]:
			current_board.push(move)
		
		next_board = current_board.copy()
		next_board.push(history[i])
		
		# Train the white model
		current_value = model.predict(np.array([board_to_array(current_board)]))[0]
		
		if i == len(history) - 1:
			next_value = get_reward(game)
		else:
			next_value = model.predict(np.array([board_to_array(next_board)]))[0]
		
		target_value = current_value + 0.95 * (next_value - current_value)
		
		model.fit(np.array([board_to_array(current_board)]), np.array([target_value]), verbose=0)
		model.save('./engines/Manual-Tensor.h5')
		with open("./engines/epsilon.txt","w",encoding='utf-8') as epsi:
			epsi.write(str(epsilon))

rep = True
while rep:
	model = load_model("./engines/Manual-Tensor.h5")

	executor = ThreadPoolExecutor(max_workers=1)
	move_queue = Queue()


	white_pieces = {
		'R': pygame.image.load('./chess_set/white/wR.png'),
		'N': pygame.image.load('./chess_set/white/wN.png'),
		'B': pygame.image.load('./chess_set/white/wB.png'),
		'Q': pygame.image.load('./chess_set/white/wQ.png'),
		'K': pygame.image.load('./chess_set/white/wK.png'),
		'P': pygame.image.load('./chess_set/white/wP.png')
	}
	black_pieces = {
		'r': pygame.image.load('./chess_set/black/bR.png'),
		'n': pygame.image.load('./chess_set/black/bN.png'),
		'b': pygame.image.load('./chess_set/black/bB.png'),
		'q': pygame.image.load('./chess_set/black/bQ.png'),
		'k': pygame.image.load('./chess_set/black/bK.png'),
		'p': pygame.image.load('./chess_set/black/bP.png')
	}
	player = display_color_choice(screen)
	fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
	board = []
	for row in fen.split('/'):
		board_row = []
		for char in row:
			if char.isdigit():
				board_row.extend([' '] * int(char))
			else:
				board_row.append(char)
		board.append(board_row)
	dragging = False
	drag_piece = None
	drag_start_square = None
	choice = "again"
	game = chess.Board()


	# Add these variables to keep track of the last move
	# Add these variables to keep track of the last move and legal moves
	# Add these variables to keep track of the last move and legal moves
	last_move_start = None
	last_move_end = None
	future = None
	is_retraining = False
	legal_moves = []

	with open("engines/epsilon.txt","r",encoding='utf-8') as ep:
		epsilon = ep.read()
	epsilon = float(epsilon)
	clock = pygame.time.Clock()
	# Call the function at the start of your game
	on = True
	while on:
		if game.is_game_over():
			mate.play()
			choice = display_end_menu(game.result(), screen.copy(), player)
			thread_retrain = Thread(target=retrain)
			thread_retrain.start()
			if choice == "again":
				pass
			else:
				rep = False
				exit()
			on = False
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				sys.exit()
			elif event.type == pygame.MOUSEBUTTONDOWN:
				x, y = pygame.mouse.get_pos()
				col = x // square_size
				row = y // square_size
				if board[row][col] != ' ':
					if player == "White":
						if game.turn == chess.WHITE and board[row][col].isupper():
							dragging = True
							drag_piece = board[row][col]
							drag_start_square = (row, col)
							board[row][col] = ' '
							legal_moves = []
							start_square = chr(col + ord('a')) + str(8 - row)
							for move in game.legal_moves:
								if move.uci()[:2] == start_square:
									end_square = move.uci()[2:]
									end_col = ord(end_square[0]) - ord('a')
									end_row = 8 - int(end_square[1])
									legal_moves.append((end_row, end_col))
					elif player == "Black":
						if game.turn == chess.BLACK and board[row][col].islower():
							dragging = True
							drag_piece = board[row][col]
							drag_start_square = (row, col)
							board[row][col] = ' '
							legal_moves = []
							start_square = chr(col + ord('a')) + str(8 - row)
							for move in game.legal_moves:
								if move.uci()[:2] == start_square:
									end_square = move.uci()[2:]
									end_col = ord(end_square[0]) - ord('a')
									end_row = 8 - int(end_square[1])
									legal_moves.append((end_row, end_col))
			elif event.type == pygame.MOUSEBUTTONUP:
				if dragging:
					x, y = pygame.mouse.get_pos()
					col = x // square_size
					row = y // square_size
					start_row, start_col = drag_start_square
					if start_row != row or start_col != col:
						start_square = chr(start_col + ord('a')) + str(8 - start_row)
						release_square = chr(col + ord('a')) + str(8 - row)
						if release_square not in generate_chess_squares():
							release_square = start_square
						uci_move = start_square + release_square
						if drag_piece == 'P':
							if start_row == 1:
								if player == "White":
									uci_move += display_promotion_options(screen, chess.WHITE)
								else:
									pass
						elif drag_piece == 'p':
							if start_row == 6:
								if player == "Black":
									uci_move += display_promotion_options(screen, chess.BLACK)
								else:
									pass
						leg = [move.uci() for move in game.legal_moves]
						if uci_move in leg:
							move_obj = chess.Move.from_uci(uci_move)
							if game.is_capture(move_obj):
								cap_sound.play()
							else:
								move_sound.play()
							game.push_uci(uci_move)
							last_move_start = (start_row, start_col)
							last_move_end = (row, col)
							fen = game.board_fen()
							board = []
							for row in fen.split('/'):
								board_row = []
								for char in row:
									if char.isdigit():
										board_row.extend([' '] * int(char))
									else:
										board_row.append(char)
								board.append(board_row)
						else:
							board[start_row][start_col] = drag_piece
					else:
						board[start_row][start_col] = drag_piece
					dragging=False
					drag_piece=None
					drag_start_square=None

		#check if a move is available from the mcts function
		if not move_queue.empty():
			future=move_queue.get()
			if future.done():
				uci_move=future.result()
				move_obj = chess.Move.from_uci(uci_move)
				if game.is_capture(move_obj):
					cap_sound.play()
				else:
					move_sound.play()
				game.push_uci(uci_move)
				last_move_start=(8-int(uci_move[3]), ord(uci_move[2])-ord('a'))
				last_move_end=(8-int(uci_move[1]), ord(uci_move[0])-ord('a'))
				fen=game.board_fen()
				board=[]

				for row in fen.split('/'):
					board_row=[]
					for char in row:
						if char.isdigit():
							board_row.extend([' ']*int(char))
						else:
							board_row.append(char)
					board.append(board_row)
		#make a new move for the black pieces
		if not game.is_game_over() and move_queue.empty() and (future is None or future.done()):
			if player == "Black":
				if game.turn == chess.WHITE:
					simulations=10 # set simulations value here
					future=executor.submit(mcts, game, model, epsilon, simulations=simulations)
					move_queue.put(future)
			if player == "White":
				if game.turn == chess.BLACK:
					simulations=10 # set simulations value here
					future=executor.submit(mcts, game, model, epsilon, simulations=simulations)
					move_queue.put(future)
		# Size of each square on your chessboard
		SQUARE_SIZE = 90  # since each square is 90x90 pixels
		screen.fill((0, 0, 0))
		for row in range(8):
			for col in range(8):
				if (row + col) % 2 == 0:
					color = white
				else:
					color = black
				if (row, col) == last_move_start or (row, col) == last_move_end:
					color = (205, 210, 106, 10)  # gold
				# highlight the king's square in red when it is in check
				if game.is_check() and (row, col) == game.king(game.turn):
					color = (255, 0, 0) # red
				x = col * square_size
				y = row * square_size
				pygame.draw.rect(screen, color, (x, y, square_size, square_size))
				piece = board[row][col]
				if piece in white_pieces:
					piece_img = white_pieces[piece]
					screen.blit(piece_img, (x + (square_size - piece_img.get_width()) // 2,
											y + (square_size - piece_img.get_height()) // 2))
				elif piece in black_pieces:
					piece_img = black_pieces[piece]
					screen.blit(piece_img, (x + (square_size - piece_img.get_width()) // 2,
											y + (square_size - piece_img.get_height()) // 2))
				if (row, col) in legal_moves and dragging:
					circle_surface = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
					pygame.draw.circle(circle_surface, (100, 111, 64, 150), (square_size // 2, square_size // 2), square_size // 6)
					screen.blit(circle_surface, (x, y))
				if drag_start_square == (row, col) and dragging:
					square_surface = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
					square_surface.fill((100, 111, 64, 150))
					screen.blit(square_surface, (x, y))
		if game.is_check():
			if game.turn == chess.BLACK:
				king_square = game.king(chess.BLACK)
			else:
				king_square = game.king(chess.WHITE)
			# Calculate pixel position
			x_position = (king_square % 8) * SQUARE_SIZE
			y_position = (7-(king_square // 8)) * SQUARE_SIZE
			
			# Draw rectangle around the square using pygame
			pygame.draw.rect(screen, (255, 0, 0), (x_position, y_position, SQUARE_SIZE, SQUARE_SIZE), 3)  # 3 is the thickness of the rectangle's border
		if dragging:
			x,y = pygame.mouse.get_pos()
			if drag_piece in white_pieces:
				piece_img = white_pieces[drag_piece]
			else:
				piece_img = black_pieces[drag_piece]
			screen.blit(piece_img,(x-piece_img.get_width()//2,y-piece_img.get_height()//2))
		pygame.display.update()
		clock.tick(30)
