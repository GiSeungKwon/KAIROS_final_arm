from pymycobot import MyCobot320
import time


mc = MyCobot320('COM3',115200)
mc.set_gripper_mode(0)
mc.init_electric_gripper()
time.sleep(2)
mc.set_electric_gripper(0)
mc.set_gripper_value(100,20,1)