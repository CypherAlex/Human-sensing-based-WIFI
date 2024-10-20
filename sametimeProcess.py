import subprocess
import threading
import time
# 打开第一个命令行窗口并运行python test.py命令
# subprocess.Popen(['start', 'cmd', '/k', 'python', 'collect.py'], shell=True)

# 打开第二个命令行窗口并运行python test.py命令
# subprocess.Popen(['start', 'cmd', '/k', 'python', 'collect2.py'], shell=True)
# outfile = open('./data/output.log','a')
import serial.tools.list_ports
def portOK(p):
    ports = serial.tools.list_ports.comports(include_links=False)
    for port in ports:
        if(p == port.device):
            return 1
    return 0

port1 = "COM7"
port2 = "COM3"
# 可选LLFT,HT-LFT,STBC-HT-LTF
csi_output_type = "all"

#每次动作持续5秒
t = "15"
time.sleep(1)
actions = [
    #"circle"
    #"horizontal"
    #"vertical"
    #"wave"
    #"walking_wang_hide"
    #"walking_xu_phone_1"
    "test"
]
for action in actions:
    # time.sleep(10)
    #循环次数
    for i in range(1,56):
    # for i in [5,49]:

        while True:
            try:
                # 尝试打开串口
                ser1 = serial.Serial(port1)
                # ser2 = serial.Serial(port2)
                ser1.close()
                # ser2.close()
                print("Serial port opened successfully")
                break  # 跳出循环
            except serial.SerialException as e:
                # time.sleep(0.1)
                pass

        p2 = subprocess.Popen(
            ['cmd', '/c','start','/high', 'python', 'multiSerials.py', "-p", port1,  "-n", str(i) ,"-t", action, "-d",t, "-o", csi_output_type],
            shell=True,creationflags=subprocess.HIGH_PRIORITY_CLASS)


        # p1.wait()
        # p2.wait()
        time.sleep(int(t)+5)
        print(f"Finished {i}!! NEXT PART!!")
