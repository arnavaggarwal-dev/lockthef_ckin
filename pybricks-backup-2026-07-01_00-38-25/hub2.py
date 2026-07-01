from pybricks.hubs import TechnicHub
from pybricks.pupdevices import Motor
from pybricks.parameters import Port
from pybricks.tools import wait

hub = TechnicHub()
motor = Motor(Port.B)

while True:
    # Get hub tilt (front/back)
    pitch = hub.imu.tilt()[0]

    # Convert tilt to motor speed
    speed = pitch * 10   # Adjust multiplier if needed

    # Limit maximum speed
    if speed > 1000:
        speed = 1000
    elif speed < -1000:
        speed = -1000

    motor.run(speed)

    wait(20)