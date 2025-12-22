import os
import time
from time import sleep
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
    "sold_out": "sold_out.png",
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


def match_template(src: np.ndarray, template: np.ndarray, threshold: float = IMAGE_MATCH_THRESHOLD) -> Optional[
    Tuple[int, int]]:
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
        bubble_y_positions = [int(0.40 * h), int(0.60 * h), int(0.75 * h)]
        click_relative(500, bubble_y_positions[index], rect)
        time.sleep(0.8)

    def purchase_items():
        note_template = load_template("note")
        hundred_template = load_template("hundred")
        buy_template = load_template("buy")
        confirm_template = load_template("confirm")
        if note_template is None and hundred_template is None:
            return

        # 先买所有 note（用 sold_out 标记过滤，避免重复点已买过/已售罄的格子）
        sold_out_template = load_template("sold_out")

        def is_near_any(x: int, y: int, points: list[tuple[int, int]], dist: int = 45) -> bool:
            # dist 可调：售罄标记和物品图标通常很近，45~70 都常用
            for px, py in points:
                if abs(x - px) + abs(y - py) <= dist:
                    return True
            return False

        while True:
            check_pause_and_running()
            img, rect = capture_emulator()

            note_positions = find_all_matches(img, note_template, 0.8) if note_template is not None else []
            if not note_positions:
                break

            # 一次性扫描所有 sold_out，作为“已售罄/已购买”的标记列表
            sold_positions = []
            if sold_out_template is not None:
                sold_positions = find_all_matches(img, sold_out_template, 0.8)

            # 过滤掉“旁边已经有 sold_out 标记”的 note
            filtered_notes = [p for p in note_positions if not is_near_any(p[0], p[1], sold_positions, dist=150)]
            if not filtered_notes:
                # 当前屏所有 note 都已经售罄/买过（或被标记了），结束 note 购买
                break

            any_bought = False

            for cx, cy in filtered_notes:
                check_pause_and_running()

                # 点 note
                img1, rect1 = capture_emulator()
                pyautogui.click(rect1[0] + cx, rect1[1] + cy)
                time.sleep(0.25)

                # 点 buy
                img2, rect2 = capture_emulator()
                buy_pos = match_template(img2, buy_template, threshold=0.8)
                if not buy_pos:
                    continue
                bx, by = buy_pos
                pyautogui.click(rect2[0] + bx, rect2[1] + by)
                time.sleep(0.35)

                # confirm（如果有）
                img3, rect3 = capture_emulator()
                conf_pos = match_template(img3, confirm_template,
                                          threshold=0.8) if confirm_template is not None else None
                if conf_pos:
                    kx, ky = conf_pos
                    pyautogui.click(rect3[0] + kx, rect3[1] + ky)
                    time.sleep(0.2)

                # 收尾点击空白
                for _ in range(20):
                    click_blank(rect3)
                    sleep(0.05)

                any_bought = True

                #  关键：买完后立刻重新截图，让 sold_out 出现/刷新，然后下一轮再过滤
                # （避免这一轮坐标列表继续点到已变化的 UI）
                break

            if not any_bought:
                # 有 note 但全都买不了，防止死循环
                print("debug: no more purchasable notes found, breaking out of loop")
                break

            time.sleep(0.2)

        # 再买所有 100，并在每次买完后走大拇指 + 拿走 + 空白（加入 sold_out 过滤）
        sold_out_template = load_template("sold_out")

        def is_near_any(x: int, y: int, points: list[tuple[int, int]], dist: int = 45) -> bool:
            for px, py in points:
                if abs(x - px) + abs(y - py) <= dist:
                    return True
            return False

        while True:
            check_pause_and_running()
            img, rect = capture_emulator()

            hundred_positions = find_all_matches(img, hundred_template, 0.9) if hundred_template is not None else []
            if not hundred_positions:
                print("debug: no more purchasable 100s found, breaking out of loop")
                break

            # 扫描当前屏所有 sold_out 标记
            sold_positions = []
            if sold_out_template is not None:
                sold_positions = find_all_matches(img, sold_out_template, 0.8)

            #  过滤掉“旁边有 sold_out 标记”的 100
            filtered_hundreds = [p for p in hundred_positions if not is_near_any(p[0], p[1], sold_positions, dist=150)]
            if not filtered_hundreds:
                print("debug: all 100s are sold out (filtered), breaking out of loop")
                break

            bought_one = False

            for cx, cy in filtered_hundreds:
                check_pause_and_running()
                img1, rect1 = capture_emulator()
                pyautogui.click(rect1[0] + cx, rect1[1] + cy)
                time.sleep(0.3)

                img2, rect2 = capture_emulator()
                buy_pos = match_template(img2, buy_template, threshold=0.8)
                if not buy_pos:
                    # 点了但没出现 buy：很可能是售罄/不可买，跳过
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
                bought_one = True

                # 买完一个就 break，下一轮重新截图重新过滤 sold_out（避免坐标重排/假命中）
                break

            if not bought_one:
                print("debug: found 100s but none could be bought (no buy button), breaking to avoid loop")
                break

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
    time.sleep(1)
    click_bubble(1)
    take_thumb_reward()
    time.sleep(1)
    click_bubble(0)
    refresh_template = load_template("refresh")
    back_template = load_template("back")
    tag_template = load_template("tag")
    purchase_items()
    time.sleep(1)
    if final_shop:
        refreshes = 0
        while refreshes < 2:
            check_pause_and_running()
            img, rect = capture_emulator()
            pos = match_template(img, refresh_template, threshold=0.8)
            if pos:
                x, y = pos
                time.sleep(0.3)
                pyautogui.click(rect[0] + x, rect[1] + y)
                time.sleep(0.1)
                purchase_items()
                refreshes += 1
            else:
                break
        if refreshes == 2:
            # 购买流程结束，退出商店，准备退出星塔
            print("debug: reached 2 refreshes in final shop, exiting shop")
            img, rect = capture_emulator()
            back_pos = match_template(img, back_template, threshold=0.8)
            if back_pos:
                time.sleep(0.5)
                click_blank(rect)
                x, y = back_pos
                pyautogui.click(rect[0] + x, rect[1] + y)
                print("debug : quit shop after 2 refreshes")
                time.sleep(0.5)

    else:
        # debug: 提醒购买流程已结束
        print("Purchase process completed. Checking for refresh and back options...")
        img, rect = capture_emulator()
        back_pos = match_template(img, back_template, threshold=0.8)
        if back_pos:
            x, y = back_pos
            pyautogui.click(rect[0] + x, rect[1] + y)
            print("debug : quit shop")
            time.sleep(0.5)
        else:
            print("debug: no back button found, pause")
            toggle_pause()
    click_bubble(2)
    if final_shop:
        # 最后一个商店，如果有确定按钮，点击它
        print("debug: final shop - checking for confirm button")
        confirm_template = load_template("confirm")
        if confirm_template is not None:
            img, rect = capture_emulator()
            confirm_pos = match_template(img, confirm_template, threshold=0.8)
            if confirm_pos:
                cx, cy = confirm_pos
                pyautogui.click(rect[0] + cx, rect[1] + cy)
                time.sleep(0.5)



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
            time.sleep(0.5)

            confirm_template = load_template("confirm")
            if confirm_template is not None:
                img2, rect2 = capture_emulator()
                conf_pos = match_template(img2, confirm_template, threshold=0.8)
                if conf_pos:
                    cx, cy = conf_pos
                    pyautogui.click(rect2[0] + cx, rect2[1] + cy)
                    time.sleep(1.5)
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

    MAX_RUNS = 7
    run_count = 0

    while RUNNING and run_count < MAX_RUNS:
        run_count += 1
        print(f"===== Run {run_count}/{MAX_RUNS} =====")
        try:
            main_loop()
            time.sleep(1.0)
        except KeyboardInterrupt:
            print("Stopped by user.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2.0)
        finally:
            SKIP_INITIAL_WAIT = False

    print("All runs completed.")

