from pybricks.hubs import TechnicHub
from pybricks.pupdevices import Motor
from pybricks.parameters import Port, Color
from pybricks.tools import wait
from usys import stdin, stdout
from uselect import poll

hub = TechnicHub()
motor_a = Motor(Port.A)
motor_b = Motor(Port.B)

keyboard = poll()
keyboard.register(stdin)
smooth_rainbow = [Color(h=hue) for hue in range(360)]
hub.light.animate(smooth_rainbow, interval=20)

LIFT_SPEED = 50  # slow, deg/s — tune this for your mechanism

while True:
    stdout.buffer.write(b"rdy")

    if keyboard.poll(10000):
        cmd = stdin.buffer.read(3)

        if cmd == b"foc":
            hub.light.on(Color.BLUE)
            motor_a.run_angle(990, 1800*2, wait=False)
            wait(7000)
            motor_a.stop()
            smooth_rainbow = [Color(h=hue) for hue in range(360)]
            hub.light.animate(smooth_rainbow, interval=20)

        elif cmd == b"up_":
            motor_b.run(-LIFT_SPEED)

        elif cmd == b"dwn":
            motor_b.run(LIFT_SPEED)

        elif cmd == b"stp":
            motor_b.stop()

        elif cmd == b"bye":
            hub.light.off()
            break