import matplotlib.pyplot as plt
import pandas as pd
import os
from io import BytesIO
import owncloud
from datetime import datetime


def main():
    public_link = 'https://rwth-aachen.sciebo.de/s/ZpRI67lfqD27VlK?path=%2F'
    folder_password = 'CIE_2021'
    oc = owncloud.Client.from_public_link(public_link, folder_password=folder_password)

    save_location = 'C:\\Users\\Neel Savla\\Desktop\\Files\\MS in Germany\\RWTH Aachen\\MME CAME\\MME-CAME_Course_Documents\\Computational Intelligence in Engineering\\Project A\\Generated_Data'
    date_time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    fullsave_location = os.path.join (save_location, date_time)
    os.makedirs(fullsave_location)

    subject_id = ['215'] #, '216', '217', '218']

    files = oc.list('/Smartphone3/', depth = 'infinity')
    for file in files:
        if not file.is_dir():
            if any(x in file.get_path() for x in subject_id) :
                filename = os.path.splitext(file.get_name())[0]
                parent = os.path.basename(os.path.normpath(file.get_path()))
                content=oc.get_file_contents(file.get_path() + '/' + file.get_name()) 
                csv = pd.read_csv(BytesIO(content))
                for dir in ['X', 'Y', 'Z']: 
                    plt.clf()
                    if filename == 'Gyroscope': 
                        plt.plot(csv['Time(s)'], csv[f'{dir}(rad/s)'])
                        plt.ylabel(f'{dir}(rad/s)')
                        plt.title('Time vs. Angular Velocity')
                    else: 
                        plt.plot(csv['Time(s)'], csv[f'{dir}(m/s^2)'])
                        plt.ylabel(f'{dir}(m/s^2)')
                        plt.title('Time vs. Acceleration')
                    plt.xlabel('Time (s)')
                    plt.savefig(os.path.join(fullsave_location, f'{parent}_{filename}_time_v._{dir}.png'))

if __name__ == '__main__':
    main()