import pygame
import sys
from eyetrackingCPT import ColorMatchingTest
import instructionsattentiontest

# Colors
WHITE = (255, 255, 255)
GRAY = (0, 0, 0)

# Initialize Pygame
pygame.init()

# Screen dimensions
screen_info = pygame.display.Info()
SCREEN_WIDTH = screen_info.current_w
SCREEN_HEIGHT = screen_info.current_h
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Pygame Menu")

# Font
font = pygame.font.Font(None, 36)

# Function to display text
def draw_text(text, font, color, surface, x, y):
    textobj = font.render(text, 1, color)
    textrect = textobj.get_rect()
    textrect.center = (x, y)
    surface.blit(textobj, textrect)

# Function to create buttons
def create_button(surface, color, x, y, width, height, text, text_color):
    pygame.draw.rect(surface, color, (x, y, width, height))
    draw_text(text, font, text_color, surface, x + width // 2, y + height // 2)

# Main menu function
def main_menu():
    while True:
        screen.fill(WHITE)
        draw_text("Eye-Tracking Attention Test", font, GRAY, screen, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4)
        create_button(screen, GRAY, SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 - 50, 200, 50, "Start Test", WHITE)
        create_button(screen, GRAY, SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 + 50, 200, 50, "Instructions", WHITE)
        create_button(screen, GRAY, SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 + 150, 200, 50, "Quit", WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if SCREEN_WIDTH // 2 - 100 <= mouse_pos[0] <= SCREEN_WIDTH // 2 + 100 and SCREEN_HEIGHT // 2 - 50 <= mouse_pos[1] <= SCREEN_HEIGHT // 2:
                    # Start Test button clicked
                    test = ColorMatchingTest()
                    test.run_test()  # Call the run_test method
                elif SCREEN_WIDTH // 2 - 100 <= mouse_pos[0] <= SCREEN_WIDTH // 2 + 100 and SCREEN_HEIGHT // 2 + 50 <= mouse_pos[1] <= SCREEN_HEIGHT // 2 + 100:
                    # Instructions button clicked
                    instructionsattentiontest.instructions_screen()
                elif SCREEN_WIDTH // 2 - 100 <= mouse_pos[0] <= SCREEN_WIDTH // 2 + 100 and SCREEN_HEIGHT // 2 + 150 <= mouse_pos[1] <= SCREEN_HEIGHT // 2 + 300:
                    # Quit button clicked
                    pygame.quit()
                    sys.exit()

        pygame.display.update()

# Run the main menu
if __name__ == "__main__":
    main_menu()
