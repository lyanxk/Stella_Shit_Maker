import os
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
import keyboard

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(BASE_DIR, "resources")

IMAGE_MATCH_THRESHOLD = 0.80

TEMPLATES = {
    "quick_start": "quick_start_button.png",
    "next": "next.png",
    "start_battle": "start_battle.png",
    "choice": "choice.png",
    "tag": "tag.png",
    "note": "note.png",
    "hundred": "100.png",
    "buy": "buy.png",
    "refresh": "refresh.png",
    "back": "back.png",
    "leave": "leave.png",
    "save": "save.png",
    "enter_shop": "enter_shop.png",
    "not_enough_money": "not_enough_money.png",
    "enter": "enter_button.png",
    "confirm": "confirm.png",
    "select": "select.png",
    "select_confirm": "select_confirm.png",
    "shop": "shop.png",
    "strengthen": "strengthen.png",
}

PAUSED = False
SKIP_INITIAL_WAIT = False
RUNNING = True


def toggle_pause():
    global PAUSED
    PAUSED = not PAUSED
    if PAUSED:
        print("[Hotkey] Paused")
    else:
        print("[Hotkey] Resumed")


def mark_skip_initial():
    global SKIP_INITIAL_WAIT
    SKIP_INITIAL_WAIT = True
    print("[Hotkey] Skip initial waits enabled")


def stop_running():
    global RUNNING
    RUNNING = False
    print("[Hotkey] Stop requested")


def check_pause_and_running():
    global RUNNING
    if not RUNNING:
        raise KeyboardInterrupt("User requested stop")
    while PAUSED and RUNNING:
        time.sleep(0.1)
    if not RUNNING:
        raise KeyboardInterrupt("User requested stop")


def load_template(name: str) -> Optional[np.ndarray]:
    filename = TEMPLATES.get(name)
    if not filename:
        return None
    path = os.path.join(RESOURCE_DIR, filename)
    if not os.path.isfile(path):
        return None
    template = cv2.imread(path, cv2.IMREAD_COLOR)
    if template is None or template.size == 0:
        return None
    return template


def get_emulator_window() -> Optional[gw.Win32Window]:
    for window in gw.getAllWindows():
        title = window.title.lower()
        if "mumu" in title or "模拟器" in title:
            return window
    return None


def capture_emulator() -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    check_pause_and_running()
    win = get_emulator_window()
    if not win:
        raise RuntimeError("MuMu window not found. Ensure the emulator is running.")
    left, top, width, height = win.left, win.top, win.width, win.height
    if width <= 0 or height <= 0:
        win.restore()
        time.sleep(0.5)
        left, top, width, height = win.left, win.top, win.width, win.height
    screenshot = pyautogui.screenshot(region=(left, top, width, height))
    img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return img, (left, top, width, height)


def match_template(src: np.ndarray, template: np.ndarray, threshold: float = IMAGE_MATCH_THRESHOLD) -> Optional[Tuple[int, int]]:
    if src is None or template is None:
        return None
    result = cv2.matchTemplate(src, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val < threshold:
        return None
    t_h, t_w = template.shape[:2]
    x, y = max_loc
    center_x = x + t_w // 2
    center_y = y + t_h // 2
    return center_x, center_y


def wait_and_click(template_name: str, timeout: float = 30.0, threshold: float = IMAGE_MATCH_THRESHOLD) -> bool:
    template = load_template(template_name)
    if template is None:
        raise ValueError(f"Template {template_name} not found in resources.")
    start = time.time()
    is_initial_btn = template_name in ("quick_start", "next", "start_battle")
    while time.time() - start < timeout:
        check_pause_and_running()
        if is_initial_btn and SKIP_INITIAL_WAIT:
            print(f"[Skip] Skip waiting for {template_name}")
            return False
        img, (left, top, width, height) = capture_emulator()
        pos = match_template(img, template, threshold)
        if pos:
            x, y = pos
            screen_x = left + x
            screen_y = top + y
            pyautogui.click(screen_x, screen_y)
            return True
        time.sleep(0.5)
    print(f"[Timeout] {template_name} not found in {timeout} seconds")
    return False


def click_relative(offset_x: int, offset_y: int, window_rect: Tuple[int, int, int, int], delay: float = 0.0):
    check_pause_and_running()
    left, top, _, _ = window_rect
    pyautogui.click(left + offset_x, top + offset_y)
    if delay:
        time.sleep(delay)


def continuous_fast_click(delay: float = 0.05, duration: float = 2.0):
    img, rect = capture_emulator()
    left, top, width, height = rect
    click_x = left + 10
    click_y = top + height // 2
    end_time = time.time() + duration
    while time.time() < end_time:
        check_pause_and_running()
        pyautogui.click(click_x, click_y)
        time.sleep(delay)


def select_choice_or_first():
    select_icon = load_template("select")
    choice_icon = load_template("choice")

    img, rect = capture_emulator()
    check_pause_and_running()

    select_pos = match_template(img, select_icon, threshold=0.7) if select_icon is not None else None
    if select_pos:
        sx, sy = select_pos
        print(f"[Debug] select matched at ({sx}, {sy})")
        pyautogui.click(rect[0] + sx, rect[1] + sy)
        time.sleep(0.3)
        t0 = time.time()
        while time.time() - t0 < 3:
            check_pause_and_running()
            img2, rect2 = capture_emulator()
            conf_icon = load_template("select_confirm")
            conf_pos = match_template(img2, conf_icon, threshold=0.7) if conf_icon is not None else None
            if conf_pos:
                cx, cy = conf_pos
                print(f"[Debug] select_confirm matched at ({cx}, {cy})")
                pyautogui.click(rect2[0] + cx, rect2[1] + cy)
                break
            time.sleep(0.2)
        return

    choice_pos = match_template(img, choice_icon, threshold=0.8) if choice_icon is not None else None
    if choice_pos:
        cx, cy = choice_pos
        print(f"[Debug] choice matched at ({cx}, {cy})")
        pyautogui.click(rect[0] + cx, rect[1] + cy)
        return

    left, top, width, height = rect
    x = left + int(width * 0.2)
    y = top + height // 2
    print(f"[Debug] fallback click at ({x}, {y})")
    pyautogui.click(x, y)

def click_blank(rect):
    check_pause_and_running()
    left, top, width, height = rect
    x = left + 10
    y = top + height // 2
    pyautogui.click(x, y)

def find_all_matches(img: np.ndarray, template: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    if img is None or template is None:
        return []
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(result >= threshold)
    if len(xs) == 0:
        return []
    t_h, t_w = template.shape[:2]
    centers = [(int(x + t_w // 2), int(y + t_h // 2)) for x, y in zip(xs, ys)]
    centers.sort(key=lambda p: (p[1], p[0]))
    filtered = []
    min_dist = 20
    for cx, cy in centers:
        ok = True
        for fx, fy in filtered:
            if abs(cx - fx) + abs(cy - fy) < min_dist:
                ok = False
                break
        if ok:
            filtered.append((cx, cy))
    return filtered


def handle_shop(final_shop: bool = False):
    def click_bubble(index: int):
        img, rect = capture_emulator()
        h = rect[3]
        bubble_y_positions = [int(0.40 * h), int(0.60 * h), int(0.80 * h)]
        click_relative(500, bubble_y_positions[index], rect)
        time.sleep(0.8)

    def purchase_items():
        note_template = load_template("note")
        hundred_template = load_template("hundred")
        buy_template = load_template("buy")
        confirm_template = load_template("confirm")
        if note_template is None and hundred_template is None:
            return

        # 先买所有 note
        while True:
            check_pause_and_running()
            img, rect = capture_emulator()
            note_positions = find_all_matches(img, note_template, 0.8) if note_template is not None else []
            if not note_positions:
                break
            for cx, cy in note_positions:
                check_pause_and_running()
                img1, rect1 = capture_emulator()
                pyautogui.click(rect1[0] + cx, rect1[1] + cy)
                time.sleep(0.3)
                img2, rect2 = capture_emulator()
                buy_pos = match_template(img2, buy_template, threshold=0.8)
                if not buy_pos:
                    continue
                bx, by = buy_pos
                pyautogui.click(rect2[0] + bx, rect2[1] + by)
                time.sleep(0.4)
                img3, rect3 = capture_emulator()
                conf_pos = match_template(img3, confirm_template,
                                          threshold=0.8) if confirm_template is not None else None
                if conf_pos:
                    kx, ky = conf_pos
                    pyautogui.click(rect3[0] + kx, rect3[1] + ky)
                    time.sleep(0.2)
                click_blank(rect3)
                click_blank(rect3)
                click_blank(rect3)
                time.sleep(0.8)
            time.sleep(0.2)

        # 再买所有 100，并在每次买完后走大拇指 + 拿走 + 空白
        while True:
            check_pause_and_running()
            img, rect = capture_emulator()
            hundred_positions = find_all_matches(img, hundred_template, 0.8) if hundred_template is not None else []
            if not hundred_positions:
                break
            for cx, cy in hundred_positions:
                check_pause_and_running()
                img1, rect1 = capture_emulator()
                pyautogui.click(rect1[0] + cx, rect1[1] + cy)
                time.sleep(0.3)
                img2, rect2 = capture_emulator()
                buy_pos = match_template(img2, buy_template, threshold=0.8)
                if not buy_pos:
                    continue
                bx, by = buy_pos
                pyautogui.click(rect2[0] + bx, rect2[1] + by)
                time.sleep(0.4)
                img3, rect3 = capture_emulator()
                conf_pos = match_template(img3, confirm_template,
                                          threshold=0.8) if confirm_template is not None else None
                if conf_pos:
                    kx, ky = conf_pos
                    pyautogui.click(rect3[0] + kx, rect3[1] + ky)
                    time.sleep(0.2)
                click_blank(rect3)
                click_blank(rect3)
                click_blank(rect3)
                time.sleep(0.8)
                take_thumb_reward()
            time.sleep(0.2)

    def take_thumb_reward(timeout: float = 6.0):
        start = time.time()
        select_icon = load_template("select")
        confirm_icon = load_template("select_confirm")
        if select_icon is None or confirm_icon is None:
            return
        while time.time() - start < timeout:
            check_pause_and_running()
            img, rect = capture_emulator()
            select_pos = match_template(img, select_icon, threshold=0.7)
            if select_pos:
                sx, sy = select_pos
                pyautogui.click(rect[0] + sx, rect[1] + sy)
                time.sleep(0.3)
                t0 = time.time()
                while time.time() - t0 < 3.0:
                    check_pause_and_running()
                    img2, rect2 = capture_emulator()
                    conf_pos = match_template(img2, confirm_icon, threshold=0.7)
                    if conf_pos:
                        cx, cy = conf_pos
                        pyautogui.click(rect2[0] + cx, rect2[1] + cy)
                        time.sleep(0.2)
                        click_blank(rect2)
                        return
                    time.sleep(0.2)
                return
            time.sleep(0.2)

    click_bubble(1)
    take_thumb_reward()
    click_bubble(1)
    take_thumb_reward()
    click_bubble(0)
    refresh_template = load_template("refresh")
    back_template = load_template("back")
    tag_template = load_template("tag")
    purchase_items()
    if final_shop:
        refreshes = 0
        while refreshes < 2:
            check_pause_and_running()
            img, rect = capture_emulator()
            pos = match_template(img, refresh_template, threshold=0.8)
            if pos:
                x, y = pos
                pyautogui.click(rect[0] + x, rect[1] + y)
                time.sleep(1.0)
                purchase_items()
                refreshes += 1
            else:
                break
        if refreshes == 2:
            while True:
                check_pause_and_running()
                img, rect = capture_emulator()
                pos = match_template(img, tag_template, threshold=0.8)
                if not pos:
                    break
                x, y = pos
                pyautogui.click(rect[0] + x, rect[1] + y)
                time.sleep(0.5)
            img, rect = capture_emulator()
            back_pos = match_template(img, back_template, threshold=0.8)
            if back_pos:
                x, y = back_pos
                pyautogui.click(rect[0] + x, rect[1] + y)
                time.sleep(0.5)
    else:
        img, rect = capture_emulator()
        back_pos = match_template(img, back_template, threshold=0.8)
        if back_pos:
            x, y = back_pos
            pyautogui.click(rect[0] + x, rect[1] + y)
            time.sleep(0.5)
    click_bubble(2)



def main_loop():
    global SKIP_INITIAL_WAIT
    print("Waiting for 快速战斗 (press S to skip initial waits, P pause, Q quit)")
    wait_and_click("quick_start", timeout=60)
    print("Waiting for 下一步")
    wait_and_click("next", timeout=60)
    print("Waiting for 开始战斗")
    wait_and_click("start_battle", timeout=60)
    SKIP_INITIAL_WAIT = False
    print("Entered tower run. Starting automation…")
    shop_counter = 0
    max_shops = 4
    while True:
        check_pause_and_running()
        continuous_fast_click(delay=0.05, duration=1.5)
        img, rect = capture_emulator()
        save_template = load_template("save")
        save_pos = match_template(img, save_template, threshold=0.8)
        if save_pos:
            sx, sy = save_pos
            pyautogui.click(rect[0] + sx, rect[1] + sy)
            print("Found 保存记录. Exiting run…")
            break
        enter_shop_template = load_template("enter_shop")
        shop_pos = match_template(img, enter_shop_template, threshold=0.8)
        if shop_pos and shop_counter < max_shops:
            print(f"Encountered shop {shop_counter + 1}")
            final_shop = shop_counter == max_shops - 1
            handle_shop(final_shop=final_shop)
            shop_counter += 1
            continue
        select_template = load_template("select")
        choice_template = load_template("choice")
        select_pos = match_template(img, select_template, threshold=0.7) if select_template is not None else None
        choice_pos = match_template(img, choice_template, threshold=0.8) if choice_template is not None else None
        if select_pos or choice_pos:
            print(f"[Debug] select_pos={select_pos}, choice_pos={choice_pos}")
            select_choice_or_first()
            continue

        time.sleep(0.2)
    print("Automation complete.")


if __name__ == "__main__":
    keyboard.add_hotkey("p", toggle_pause)
    keyboard.add_hotkey("s", mark_skip_initial)
    keyboard.add_hotkey("q", stop_running)
    print("Hotkeys: P=pause/resume, S=skip initial waits, Q=quit")
    try:
        main_loop()
    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        RUNNING = False
        print("Script finished.")
