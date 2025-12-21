from datetime import timedelta, datetime
from threading import Thread
import csv 
import openpyxl 
import os
import time
import concurrent.futures

folder_path = r'D:\! DLC Output\Analyzed\H\Single_Animal'




def csv_to_excel(csv_file):
    csv_path = os.path.join(folder_path, csv_file)
    
    workbook = openpyxl.Workbook()
    sheet = workbook.active

    with open(csv_path, 'r') as f:
        csv_reader = csv.reader(f)  
        for row in csv_reader:
            sheet.append(row)

    excel_path = os.path.splitext(csv_path)[0] + '.xlsx'
    workbook.save(excel_path)

# threads = []

# for csv_file in os.listdir(folder_path):
#     if csv_file.endswith('.csv'):
#         csv_to_excel(csv_file=csv_file)
        # t = Thread(target=csv_to_excel, args=[csv_file])
        # t.start()
        # threads.append(t)

# for t in threads:
#   t.join() 

def convert_timedelta(tdelta):
    days, seconds = tdelta.days, tdelta.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = (seconds % 60)
    return days, hours, minutes, seconds

def convert_all():
    start_time = time.time()

    # Get list of all video files
    files = [os.path.join(folder_path, f) 
            for f in os.listdir(folder_path) if f.endswith('DLC_resnet50_MSA - V1Mar3shuffle1_1030000.csv')]

    print(f'Number of files: {len(files)}')

    if len(files) == 0:
        print(f'No DLC output in {folder_path}')

    # Run using concurrent.futures
    print(f'Running conversion of {folder_path} with {os.cpu_count()-8} processes', flush=True)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(csv_to_excel, files)

    end_time = time.time()
    total_time = end_time - start_time

    d, h, m, s = convert_timedelta(timedelta(seconds=(time.time() - start_time)))
    print(f'Finished conversion of {folder_path} in {h}h, {m}m, {s}s', flush=True)

if __name__ == '__main__':
    convert_all()