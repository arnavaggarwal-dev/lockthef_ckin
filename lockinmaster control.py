import asyncio
from contextlib import suppress
from bleak import BleakScanner, BleakClient

PYBRICKS_CHAR_UUID = "c5f50002-8280-46da-89f4-6d8051e4aeef"
HUB_NAME = "Pybricks Hub"

async def main():
    main_task = asyncio.current_task()

    def handle_disconnect(_):
        print("Hub disconnected.")
        if not main_task.done():
            main_task.cancel()

    ready_event = asyncio.Event()

    def handle_rx(_, data: bytearray):
        if data[0] == 0x01:
            payload = data[1:]
            if payload == b"rdy":
                print("Hub is ready.")
                ready_event.set()

    print(f"Scanning for '{HUB_NAME}'...")
    device = await BleakScanner.find_device_by_name(HUB_NAME, timeout=15.0)
    if device is None:
        print("Hub not found.")
        return

    async with BleakClient(device, handle_disconnect) as client:
        await client.start_notify(PYBRICKS_CHAR_UUID, handle_rx)

        async def send(data: bytes):
            ready_event.clear()
            await client.write_gatt_char(
                PYBRICKS_CHAR_UUID,
                b"\x06" + data,
                response=True
            )
            # Wait for next rdy from hub
            await asyncio.wait_for(ready_event.wait(), timeout=15.0)

        print("Press hub button to start program...\n")

        # Wait for first rdy
        await asyncio.wait_for(ready_event.wait(), timeout=30.0)
        print("Hub ready! Type 'focus' to spin Motor A.\n")

        while True:
            cmd = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input("> ").strip().lower()
            )
            if cmd == "focus":
                await send(b"foc")
                print("Done.")
            elif cmd == "quit":
                await send(b"bye")
                break
            else:
                print("Try 'focus' or 'quit'.")

if __name__ == "__main__":
    with suppress(asyncio.CancelledError):
        asyncio.run(main())