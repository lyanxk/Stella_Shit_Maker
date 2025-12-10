"""
auto_tower.py
================

This module contains a fully automated script for quickly clearing the tower
gameplay loop in Star Tower Traveller (星塔旅人) when running through
the MuMu Android emulator.  The script makes heavy use of image
recognition to detect buttons and other UI elements on screen and then
simulates mouse clicks using the ``pyautogui`` library.  The
templates used for matching live UI state live in the ``resources``
directory alongside this file.  They are small, cropped PNG files for
each button/icon described in the project documentation.

The high‑level flow implemented here is a direct translation of the
step‑by‑step procedure provided by the user:

1.  Wait for the **快速战斗** (quick fight) button and click it.
2.  Wait for the **下一步** (next) button and click it.
3.  Wait for the **开始战斗** (start battle) button and click it.
4.  Enter the tower climb loop:
   * Repeatedly click near the left edge of the MuMu window to fast‑forward
     through empty turns.
   * If a choice dialog appears, prefer options marked with the “thumbs
     up” (choice.png) icon; otherwise choose the first option.
   * Special handling for shops: if the game reports that you have
     encountered a shop, a three‑bubble dialog appears.  The script
     chooses the second bubble twice to purchase two items, then the
     first bubble to enter the shop.  Inside the shop it buys all
     “note” items (之音) and any 100‑point drinks, optionally using
     the refresh button to reset the shop up to two times in the final
     shop.
   * When no more interactions are available (no thumbs‑up icons
     detected) the script clicks the “离开星塔” (leave tower) button to
     finish the run.

Because the emulator window can be moved or resized, the script
determines its screen coordinates at runtime using the ``pygetwindow``
package.  All click operations are expressed relative to this window.

To run this script you will need the following third‑party Python
packages installed in your environment:

* pyautogui — for taking screenshots and sending mouse clicks.
* opencv‑python — for template matching of UI elements.
* numpy — used by OpenCV.
* pillow — for basic image manipulations (installed implicitly with
  pyautogui).
* pygetwindow — for locating the MuMu window.

Note: The script does not attempt to handle every failure case.  If
images cannot be found (e.g. because the game UI changed) then the
script will time out and exit.  You may need to adjust the
``IMAGE_MATCH_THRESHOLD`` or provide updated templates in the
``resources`` directory if the game UI changes in the future.
"""

import os
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import pyautogui
import pygetwindow as gw


# Path to the directory containing this script.  Resource images are
# stored relative to this path in the ``resources`` subdirectory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(BASE_DIR, "resources")

# Confidence threshold for template matching.  If your templates are
# particularly small or low contrast you may need to lower this a
# little (e.g. 0.7) or adjust it upwards for stricter matching.
IMAGE_MATCH_THRESHOLD = 0.8

# Names of the template files used for each UI action.  These PNG
# images must exist in the ``resources`` directory.
TEMPLATES = {
    "quick_start": "quick_start_button.png",
    "next": "next.png",
    "start_battle": "start_battle.png",
    "choice": "choice.png",        # thumbs up icon for best choices
    "tag": "tag.png",              # speech bubble icon for generic options
    "note": "note.png",            # icon indicating 之音 (note) items
    "hundred": "100.png",          # text “100” for 100‑point drinks
    "buy": "buy.png",             # “购买” button in shop
    "refresh": "refresh.png",     # “刷新” button in shop
    "back": "back.png",           # generic “返回” arrow
    "leave": "leave.png",         # bubble option used to leave a floor
    "save": "save.png",           # “保存记录” button when leaving
    "enter_shop": "enter_shop.png",  # text indicating you found a shop
    "not_enough_money": "not_enough_money.png",  # insufficient currency notice
    "enter": "enter_button.png",
    "confirm": "confirm.png",
    "select": "select.png",
    "select_confirm": "select_confirm.png",
    "shop": "shop.png",
    "strengthen": "strengthen.png",
}


def load_template(name: str) -> Optional[np.ndarray]:
    filename = TEMPLATES.get(name)
    if not filename:
        return None
    path = os.path.join(RESOURCE_DIR, filename)
    if not os.path.isfile(path):
        return None

    # 统一读成 3 通道 BGR，避免和截图类型不一致
    template = cv2.imread(path, cv2.IMREAD_COLOR)
    if template is None or template.size == 0:
        return None
    return template


def get_emulator_window() -> Optional[gw.Win32Window]:
    """Find the first window whose title contains 'mumu' or '模拟器'.

    Returns None if no such window can be found.
    """
    for window in gw.getAllWindows():
        title = window.title.lower()
        if "mumu" in title or "模拟器" in title:
            return window
    return None


def capture_emulator() -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Capture a screenshot of the MuMu emulator window.

    Returns both the captured BGR image and the (left, top, width,
    height) of the window on screen.  Raises RuntimeError if the
    emulator window cannot be found.
    """
    win = get_emulator_window()
    if not win:
        raise RuntimeError("MuMu window not found. Ensure the emulator is running.")
    left, top, width, height = win.left, win.top, win.width, win.height
    # Avoid capturing when the window is minimized
    if width <= 0 or height <= 0:
        win.restore()
        time.sleep(0.5)
        left, top, width, height = win.left, win.top, win.width, win.height
    screenshot = pyautogui.screenshot(region=(left, top, width, height))
    img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return img, (left, top, width, height)


def match_template(src: np.ndarray, template: np.ndarray, threshold: float = IMAGE_MATCH_THRESHOLD) -> Optional[Tuple[int, int]]:
    """Search for a template inside a larger BGR image.

    If found with confidence >= threshold, return the centre coordinates
    (x, y) relative to ``src``.  Otherwise return None.
    """
    if src is None or template is None:
        return None
    result = cv2.matchTemplate(src, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if max_val < threshold:
        return None
    t_h, t_w = template.shape[:2]
    x, y = max_loc
    center_x = x + t_w // 2
    center_y = y + t_h // 2
    return center_x, center_y


def wait_and_click(template_name: str, timeout: float = 30.0, threshold: float = IMAGE_MATCH_THRESHOLD) -> bool:
    """Wait until a template appears on screen and click its centre.

    Args:
        template_name: Key in the TEMPLATES dictionary.
        timeout: Maximum time to wait in seconds.
        threshold: Matching confidence threshold.

    Returns:
        True if the template was found and clicked, False on timeout.
    """
    template = load_template(template_name)
    if template is None:
        raise ValueError(f"Template {template_name} not found in resources.")
    start = time.time()
    while time.time() - start < timeout:
        img, (left, top, width, height) = capture_emulator()
        pos = match_template(img, template, threshold)
        if pos:
            x, y = pos
            screen_x = left + x
            screen_y = top + y
            pyautogui.click(screen_x, screen_y)
            return True
        time.sleep(0.5)
    return False


def click_relative(offset_x: int, offset_y: int, window_rect: Tuple[int, int, int, int], delay: float = 0.0):
    """Click at a point relative to the emulator window.

    Args:
        offset_x, offset_y: Coordinates relative to the top‑left of the emulator window.
        window_rect: The (left, top, width, height) of the emulator window.
        delay: Optional sleep after the click.
    """
    left, top, _, _ = window_rect
    pyautogui.click(left + offset_x, top + offset_y)
    if delay:
        time.sleep(delay)


def continuous_fast_click(delay: float = 0.05, duration: float = 2.0):
    """Continuously click near the left centre of the emulator for a duration.

    This helper speeds through animations.  It captures the emulator
    window once to compute the click position then repeatedly clicks
    until the duration has elapsed.
    """
    img, rect = capture_emulator()
    left, top, width, height = rect
    click_x = left + 10  # 10px offset from left edge
    click_y = top + height // 2
    end_time = time.time() + duration
    while time.time() < end_time:
        pyautogui.click(click_x, click_y)
        time.sleep(delay)


def select_choice_or_first():
    """Handle generic choice dialogs.

    This function searches for the thumbs‑up (choice) icon on screen.  If
    found, it clicks the option associated with it (by clicking the
    centre of that icon).  Otherwise, it assumes a standard three‑line
    dialog and clicks on the first option (the uppermost line).  This
    relies on the choice icon being the leftmost element in each option.
    """
    # Load icons
    choice_icon = load_template("choice")
    tag_icon = load_template("tag")
    img, rect = capture_emulator()
    # Try to find the thumbs‑up icon first
    pos = match_template(img, choice_icon)
    if pos:
        click_x, click_y = pos
        pyautogui.click(rect[0] + click_x, rect[1] + click_y)
        return
    # Otherwise click the first tag bubble if present
    pos = match_template(img, tag_icon)
    if pos:
        click_x, click_y = pos
        pyautogui.click(rect[0] + click_x, rect[1] + click_y)
        return
    # Fallback: click near the top of the dialog area (safe default)
    # This uses an offset relative to the emulator window.
    click_relative(200, 350, rect)  # adjust these offsets if needed


def handle_shop(final_shop: bool = False):
    """Enter the shop and perform purchases according to the rules.

    For the first three shops, the strategy is:
    - Select the second option twice to purchase two items.
    - Then select the first option to enter the shop.
    - Inside the shop, purchase all items matching either the note
      icon or 100‑point drinks.
    - Exit the shop and select the third bubble to move on.

    For the final shop (``final_shop=True``), the strategy is the
    above plus two refreshes.  After buying all note/100‑point items
    once, refresh and repeat the buy.  On the second refresh, select
    all options with the tag icon.  When finished, exit the shop and
    click the second bubble repeatedly until no choices appear.  Then
    continue as normal.
    """
    # Helper to enter second bubble twice then first
    def click_bubble(index: int):
        """Click the bubble at a given index (0=first, 1=second, 2=third)."""
        img, rect = capture_emulator()
        h = rect[3]
        # Approximate vertical positions for the three bubbles relative to the window height
        bubble_y_positions = [int(0.70 * h), int(0.80 * h), int(0.90 * h)]
        click_relative(300, bubble_y_positions[index], rect)
        time.sleep(0.8)

    # Buy function inside shop: purchase note items and 100 drinks
    def purchase_items():
        while True:
            img, rect = capture_emulator()
            # Check if the "buy" button is present (indicating an item is selected)
            buy_template = load_template("buy")
            buy_pos = match_template(img, buy_template, threshold=0.8)
            if buy_pos:
                bx, by = buy_pos
                pyautogui.click(rect[0] + bx, rect[1] + by)
                time.sleep(0.5)
                # confirm purchase if confirm dialog appears
                confirm_template = load_template("confirm")
                img2, rect2 = capture_emulator()
                conf_pos = match_template(img2, confirm_template, threshold=0.8)
                if conf_pos:
                    cx, cy = conf_pos
                    pyautogui.click(rect2[0] + cx, rect2[1] + cy)
                time.sleep(1.0)
                continue
            # Look for note or 100 icons
            note_template = load_template("note")
            hundred_template = load_template("hundred")
            note_pos = match_template(img, note_template, threshold=0.8)
            hundred_pos = match_template(img, hundred_template, threshold=0.8)
            if note_pos:
                x, y = note_pos
                pyautogui.click(rect[0] + x, rect[1] + y)
                time.sleep(0.5)
                continue
            if hundred_pos:
                x, y = hundred_pos
                pyautogui.click(rect[0] + x, rect[1] + y)
                time.sleep(0.5)
                continue
            # If no items found, break
            break

    # Pre‑shop: click second bubble twice then first bubble
    click_bubble(1)
    click_bubble(1)
    click_bubble(0)
    # Inside shop: buy items, handle refreshes if final shop
    refresh_template = load_template("refresh")
    back_template = load_template("back")
    tag_template = load_template("tag")
    # First purchase
    purchase_items()
    if final_shop:
        # Up to two refresh cycles
        refreshes = 0
        while refreshes < 2:
            img, rect = capture_emulator()
            pos = match_template(img, refresh_template, threshold=0.8)
            if pos:
                x, y = pos
                pyautogui.click(rect[0] + x, rect[1] + y)
                time.sleep(1.0)
                # After refresh, purchase note/100 again
                purchase_items()
                refreshes += 1
            else:
                break
        # On second refresh, select all tag options
        if refreshes == 2:
            while True:
                img, rect = capture_emulator()
                pos = match_template(img, tag_template, threshold=0.8)
                if not pos:
                    break
                x, y = pos
                pyautogui.click(rect[0] + x, rect[1] + y)
                time.sleep(0.5)
            # exit shop
            img, rect = capture_emulator()
            back_pos = match_template(img, back_template, threshold=0.8)
            if back_pos:
                x, y = back_pos
                pyautogui.click(rect[0] + x, rect[1] + y)
                time.sleep(0.5)
    else:
        # not final shop: exit via back arrow when done
        img, rect = capture_emulator()
        back_pos = match_template(img, back_template, threshold=0.8)
        if back_pos:
            x, y = back_pos
            pyautogui.click(rect[0] + x, rect[1] + y)
            time.sleep(0.5)
    # After returning to bubbles: click third bubble to leave floor
    click_bubble(2)


def main_loop():
    """Execute the automated tower clear flow from beginning to end."""
    # Step 1: click quick start, next, start battle
    print("Waiting for 快速战斗 button…")
    wait_and_click("quick_start", timeout=60)
    print("Waiting for 下一步 button…")
    wait_and_click("next", timeout=60)
    print("Waiting for 开始战斗 button…")
    wait_and_click("start_battle", timeout=60)
    print("Entered tower run. Starting automation…")
    shop_counter = 0
    max_shops = 4
    while True:
        # Fast click left side to progress until something pops up
        continuous_fast_click(delay=0.05, duration=1.5)
        # Check for leave option (save button) after finishing a floor
        img, rect = capture_emulator()
        save_template = load_template("save")
        save_pos = match_template(img, save_template, threshold=0.8)
        if save_pos:
            # Click save and exit: run complete
            sx, sy = save_pos
            pyautogui.click(rect[0] + sx, rect[1] + sy)
            print("Found 保存记录. Exiting run…")
            break
        # Check for shop notification (enter_shop text)
        enter_shop_template = load_template("enter_shop")
        shop_pos = match_template(img, enter_shop_template, threshold=0.8)
        if shop_pos and shop_counter < max_shops:
            print(f"Encountered shop {shop_counter + 1}")
            final_shop = (shop_counter == max_shops - 1)
            handle_shop(final_shop=final_shop)
            shop_counter += 1
            continue
        # Check for choice/dialog bubble (tag icon) or thumbs up
        choice_template = load_template("choice")
        tag_template = load_template("tag")
        choice_pos = match_template(img, choice_template, threshold=0.8)
        tag_pos = match_template(img, tag_template, threshold=0.8)
        if choice_pos or tag_pos:
            select_choice_or_first()
            continue
        # Fallback: continue clicking; nothing to do this loop
        time.sleep(0.2)
    print("Automation complete.")


if __name__ == "__main__":
    try:
        main_loop()
    except Exception as e:
        print(f"Error: {e}")