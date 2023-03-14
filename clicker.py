import pyautogui
import keyboard

def clicker():
    while True:
        pyautogui.click()
        pyautogui.sleep(0.1)
        if keyboard.is_pressed('q'):
            break

if __name__ == "__main__":
    keyboard.add_hotkey('esc', clicker)
    keyboard.wait('esc')
