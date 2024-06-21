import pygame
import sys

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
pygame.display.set_caption("Instructions - Attention Test")

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


# Instructions screen function
def instructions_screen():
    while True:
        screen.fill(WHITE)
        draw_text("Instructions", font, GRAY, screen, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4)

        # Write your instructions here
        instructions_text = [
            "Welcome to the Attention Test!",
            "There will be a short calibration step before the test starts",
            "During the test, blue and red squares will appear on either the left or right side of the screen",
            "Press the space bar as fast as you can when a red square appears ",
            "When a blue square appears, you must look directly at the blue square, as demonstrated during calibration",
            "Green squares will appear in the centre to indicate that you have pressed the button or looked at the square correctly"
        ]

        y_offset = SCREEN_HEIGHT // 3
        for text in instructions_text:
            draw_text(text, font, GRAY, screen, SCREEN_WIDTH // 2, y_offset)
            y_offset += 50

        create_button(screen, GRAY, SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT - 150, 200, 50, "Return to Menu", WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if SCREEN_WIDTH // 2 - 100 <= mouse_pos[0] <= SCREEN_WIDTH // 2 + 100 and SCREEN_HEIGHT - 150 <= \
                        mouse_pos[1] <= SCREEN_HEIGHT - 100:
                    return  # Return to the main menu

        pygame.display.update()


# Run instructions screen
if __name__ == "__main__":
    instructions_screen()
