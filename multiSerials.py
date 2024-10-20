import sys
import csv
import serial
import pandas as pd
from os import path
from os import mkdir
from io import StringIO
import base64
import time
from datetime import datetime
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description="Read CSI data from serial port and display it graphically")
parser.add_argument('-p', '--port', dest='port', action='store', required=True,
                    help="Serial port number of csv_recv device")
parser.add_argument('-n', '--number', dest='collect_number', action='store', required=True,
                    help="collect_number")
parser.add_argument('-t', '--taget', dest='collect_tagets', action='store', required=True,
                    help="collect_tagets")
parser.add_argument('-d', '--duration', dest='collect_duration', action='store', required=True,
                    help="collect_duration")
parser.add_argument('-o', '--outtype', dest='csi_out_type', action='store', required=True,
                    help="outtype")


args = parser.parse_args()
port = args.port
N = args.collect_number
action = args.collect_tagets
t = args.collect_duration
csi_output_type = args.csi_out_type


CSI_DATA_COLUMNS_NAMES = ["type", "seq", "timestamp", "taget_seq", "taget", "mac", "rssi", "rate", "sig_mode",
                          "mcs",
                          "cwb", "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding", "sgi",
                          "noise_floor",
                          "ampdu_cnt", "channel_primary", "channel_secondary", "local_timestamp", "ant", "sig_len",
                          "rx_state", "len", "first_word_invalid", "data", "timeSever"]

def base64_decode_bin(str_data):
    try:
        bin_data = base64.b64decode(str_data)
    except Exception as e:
        print(f"Exception: {e}, data: {str_data}")

    list_data = list(bin_data)

    for i in range(len(list_data)):
        if list_data[i] > 127:
            list_data[i] = list_data[i] - 256

    return list_data

def base64_encode_bin(list_data):
    for i in range(len(list_data)):
        if list_data[i] < 0:
            list_data[i] = 256 + list_data[i]
    # print(list_data)

    str_data = "test"
    try:
        str_data = base64.b64encode(bytes(list_data)).decode('utf-8')
    except Exception as e:
        print(f"Exception: {e}, data: {list_data}")
    return str_data

def get_csi(csi_write,num):
    print("start!!!")
    progress = tqdm(total=num+1, desc="Processing")
    while num >= 0:
        # print(i)
        strings = str(ser.readline())
        strings = strings.lstrip('b\'').rstrip('\\r\\n\'')
        # print(strings)
        index = strings.find('CSI_DATA')
        if index >= 0:
            strings = strings[index:]
            csv_reader = csv.reader(StringIO(strings))
            data = next(csv_reader)
            data.append(datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S.%f')[:-3])
            # print(data)
            data_series = pd.Series(data, index=CSI_DATA_COLUMNS_NAMES)
            data_series['data'] = base64_decode_bin(data_series['data'])
            csi_write.writerow(data_series.astype(str))
            progress.update(1)
            num = num-1
    print("end!!!")

ser = serial.Serial(port=port, baudrate=2000000,
                    bytesize=8, parity='N', stopbits=1, timeout=0.1)
ser.write(('restart' + '\n').encode())

print('1秒后开始...')
time.sleep(1)

ser.write(('radar --csi_output_type ' + csi_output_type + ' --csi_output_format base64' + '\n').encode())
# ser.write(('radar --csi_output_format base64' + '\n').encode())
ser.write(('wifi_config --ssid HUAWEI-MT4D7H --password 61126112' + '\n').encode())

folder_list = ['data']
for folder in folder_list:
    if not path.exists(folder):
        mkdir(folder)
    if not path.exists(path.join(folder,action)):
        mkdir(path.join(folder,action))
csi_file = open(f'data/{action}/csi_data_{port}_{csi_output_type}_{N}.csv', 'w+')
csi_write = csv.writer(csi_file)
csi_write.writerow(CSI_DATA_COLUMNS_NAMES)
get_csi(csi_write,int(t)*100)
ser.write(('exit'+'\n').encode())
ser.close()
sys.exit(0)




