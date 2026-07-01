import asyncio
import threading
import cv2
import mediapipe as mp
import numpy as np
from bleak import BleakScanner, BleakClient

# ── Toggles ──────────────────────────────────────────────
SHOW_CAMERA     = True
VISUALISER_MODE = "eyes"  # "full" / "contours" / "eyes"

# ── BLE config ───────────────────────────────────────────
PYBRICKS_CHAR_UUID = "c5f50002-8280-46da-89f4-6d8051e4aeef"
HUB_NAME = "Pybricks Hub"

# ── Eye landmark indices (16-point full contour) ─────────
LEFT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33,  7,   163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# ── EAR settings ─────────────────────────────────────────
EAR_THRESHOLD = 0.20
CLOSED_FRAMES = 10

BTN_UP   = (20, 160, 140, 200)
BTN_DOWN = (20, 210, 140, 250)

def ear(landmarks, eye_pts, w, h):
    """EAR using 16-point full eye contour — more accurate than 6-point version."""
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in eye_pts])

    hor = np.linalg.norm(pts[0] - pts[8])

    v1 = np.linalg.norm(pts[1]  - pts[15])
    v2 = np.linalg.norm(pts[2]  - pts[14])
    v3 = np.linalg.norm(pts[3]  - pts[13])
    v4 = np.linalg.norm(pts[4]  - pts[12])
    v5 = np.linalg.norm(pts[5]  - pts[11])
    v6 = np.linalg.norm(pts[6]  - pts[10])

    return (v1 + v2 + v3 + v4 + v5 + v6) / (6.0 * hor)

def draw_eye_landmarks(frame, landmarks, eye_pts, w, h, color):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_pts]
    for pt in pts:
        cv2.circle(frame, pt, 2, color, -1)
    for i in range(len(pts)):
        cv2.line(frame, pts[i], pts[(i + 1) % len(pts)], color, 1)

def point_in_box(x, y, box):
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def draw_button(frame, box, label, active):
    x1, y1, x2, y2 = box
    color = (0, 165, 255) if active else (60, 60, 60)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    tx = x1 + (x2 - x1 - text_size[0]) // 2
    ty = y1 + (y2 - y1 + text_size[1]) // 2
    cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

async def main():
    main_task = asyncio.current_task()

    print(f"Scanning for '{HUB_NAME}'...")
    device = await BleakScanner.find_device_by_name(HUB_NAME, timeout=15.0)
    if device is None:
        print("Hub not found.")
        return

    ready_event = asyncio.Event()
    motor_busy  = False

    state = {"up_held": False, "down_held": False, "lift_cmd_sent": None}

    def handle_disconnect(_):
        print("Hub disconnected.")
        if not main_task.done():
            main_task.cancel()

    def handle_rx(_, data: bytearray):
        if data[0] == 0x01 and data[1:] == b"rdy":
            ready_event.set()

    async with BleakClient(device, handle_disconnect) as client:
        await client.start_notify(PYBRICKS_CHAR_UUID, handle_rx)

        async def send_raw(cmd: bytes):
            try:
                await client.write_gatt_char(PYBRICKS_CHAR_UUID, b"\x06" + cmd, response=True)
            except Exception as e:
                print("Send error:", e)

        async def send_focus():
            nonlocal motor_busy
            if motor_busy:
                return
            motor_busy = True
            ready_event.clear()
            await client.write_gatt_char(PYBRICKS_CHAR_UUID, b"\x06" + b"foc", response=True)
            try:
                await asyncio.wait_for(ready_event.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                pass
            motor_busy = False

        print("Press hub button to start program...")
        await asyncio.wait_for(ready_event.wait(), timeout=30.0)
        print(f"Hub ready! [camera={'on' if SHOW_CAMERA else 'off'}, mode={VISUALISER_MODE}]\n")

        loop = asyncio.get_event_loop()

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if point_in_box(x, y, BTN_UP):
                    state["up_held"] = True
                elif point_in_box(x, y, BTN_DOWN):
                    state["down_held"] = True
            elif event == cv2.EVENT_LBUTTONUP:
                state["up_held"] = False
                state["down_held"] = False

        if SHOW_CAMERA:
            cv2.namedWindow("Eye Focus Detector")
            cv2.setWindowProperty("Eye Focus Detector", cv2.WND_PROP_TOPMOST, 1)
            cv2.setMouseCallback("Eye Focus Detector", on_mouse)

        mp_face   = mp.solutions.face_mesh
        mp_draw   = mp.solutions.drawing_utils
        mp_styles = mp.solutions.drawing_styles
        use_refine = VISUALISER_MODE == "full"

        face_mesh = mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=use_refine,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        cap = cv2.VideoCapture(0)
        closed_counter = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            status  = "No face"
            avg_ear = 0.0

            if results.multi_face_landmarks:
                face_lm = results.multi_face_landmarks[0]
                lm      = face_lm.landmark

                left_ear  = ear(lm, LEFT_EYE,  w, h)
                right_ear = ear(lm, RIGHT_EYE, w, h)
                avg_ear   = (left_ear + right_ear) / 2.0

                if avg_ear < EAR_THRESHOLD:
                    closed_counter += 1
                    status = f"CLOSED ({closed_counter}/{CLOSED_FRAMES})"
                    if closed_counter == CLOSED_FRAMES and not motor_busy:
                        print("Eyes closed — sending focus!")
                        loop.create_task(send_focus())
                else:
                    closed_counter = 0
                    status = "Open"

                if SHOW_CAMERA:
                    eye_color = (0, 0, 255) if avg_ear < EAR_THRESHOLD else (0, 255, 0)
                    if VISUALISER_MODE == "full":
                        mp_draw.draw_landmarks(frame, face_lm, mp_face.FACEMESH_TESSELATION,
                            None, mp_styles.get_default_face_mesh_tesselation_style())
                        mp_draw.draw_landmarks(frame, face_lm, mp_face.FACEMESH_CONTOURS,
                            None, mp_styles.get_default_face_mesh_contours_style())
                        draw_eye_landmarks(frame, lm, LEFT_EYE,  w, h, eye_color)
                        draw_eye_landmarks(frame, lm, RIGHT_EYE, w, h, eye_color)
                    elif VISUALISER_MODE == "contours":
                        mp_draw.draw_landmarks(frame, face_lm, mp_face.FACEMESH_CONTOURS,
                            None, mp_styles.get_default_face_mesh_contours_style())
                        draw_eye_landmarks(frame, lm, LEFT_EYE,  w, h, eye_color)
                        draw_eye_landmarks(frame, lm, RIGHT_EYE, w, h, eye_color)
                    elif VISUALISER_MODE == "eyes":
                        draw_eye_landmarks(frame, lm, LEFT_EYE,  w, h, eye_color)
                        draw_eye_landmarks(frame, lm, RIGHT_EYE, w, h, eye_color)

            # ── Motor B jog control ───────────────────────
            if state["up_held"] and state["lift_cmd_sent"] != "up":
                loop.create_task(send_raw(b"up_"))
                state["lift_cmd_sent"] = "up"
            elif state["down_held"] and state["lift_cmd_sent"] != "down":
                loop.create_task(send_raw(b"dwn"))
                state["lift_cmd_sent"] = "down"
            elif not state["up_held"] and not state["down_held"] and state["lift_cmd_sent"] is not None:
                loop.create_task(send_raw(b"stp"))
                state["lift_cmd_sent"] = None

            if SHOW_CAMERA:
                color = (0, 0, 255) if closed_counter >= CLOSED_FRAMES else (0, 255, 0)
                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"Eye: {status}",      (20, 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                if motor_busy:
                    cv2.putText(frame, "MOTOR A RUNNING", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

                draw_button(frame, BTN_UP,   "UP",   state["up_held"])
                draw_button(frame, BTN_DOWN, "DOWN", state["down_held"])

                cv2.imshow("Eye Focus Detector", frame)

            await asyncio.sleep(0.01)

            if SHOW_CAMERA and cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if state["lift_cmd_sent"] is not None:
            await send_raw(b"stp")

        cap.release()
        if SHOW_CAMERA:
            cv2.destroyAllWindows()
        face_mesh.close()

if __name__ == "__main__":
    def run_async():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(main())
        finally:
            loop.close()

    t = threading.Thread(target=run_async)
    t.start()
    t.join()