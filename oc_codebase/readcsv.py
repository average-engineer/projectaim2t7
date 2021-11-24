import matplotlib.pyplot as plt
import pandas as pd
import owncloud
from io import BytesIO

public_link = 'https://rwth-aachen.sciebo.de/s/ZpRI67lfqD27VlK?path=%2F'
folder_password = 'CIE_2021'

oc = owncloud.Client.from_public_link(public_link, folder_password=folder_password)
content=oc.get_file_contents('/Smartphone3/subject215_downstairs01/Accelerometer.csv') 

#s = str(content, 'utf-8')
#buffer = StringIO(s)
csv = pd.read_csv(BytesIO(content))

#print(csv.keys())
#time = csv['Time(s)']
#print(time)
#data = pd.DataFrame({'time' : csv['Time(s)'], 'x' : csv['X(m/s^2)']})

subject_id = 215
filename = f'subject{subject_id}'

for dir in ['X', 'Y', 'Z']: 
    plt.clf()
    plt.plot(csv['Time(s)'], csv[f'{dir}(m/s^2)'])
    plt.title('Time vs. Acceleration')
    plt.xlabel('Time (s)')
    plt.ylabel(f'{dir}(m/s^2)')
    plt.savefig(f'{filename} time v. {dir}.png')


#print(content)