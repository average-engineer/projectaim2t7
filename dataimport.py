import matplotlib.pyplot as plt
import pandas as pd
import owncloud

public_link = 'https://rwth-aachen.sciebo.de/s/ZpRI67lfqD27VlK?path=%2F'
folder_password = 'CIE_2021'

oc = owncloud.Client.from_public_link(public_link, folder_password=folder_password)
oc.get_file('/anthro_150-154.xlsx', '/home/raj/Documents/modules/computational-intelligence-in-engineering/projectaim2t7/anthro_150-154.xlsx')
#df = pd.read_csv("/home/raj/Documents/modules/computational-intelligence-in-engineering/projectaim2t7/subject218_downstairs01/Accelerometer.csv", sep=';')
#print(df.head())
# xaxis = df['Time(s)'].head(100)
# yaxis = df['X(m/s^2)'].head(100)
# print()
# #plt.plot(range(0, 30))
# ##Initial axes limits are 0, 10

# # scale_factor = 2

# # xmin, xmax = plt.xlim()
# # ymin, ymax = plt.ylim()

# # plt.xlim(xmin * scale_factor, xmax * scale_factor)
# # plt.ylim(ymin * scale_factor, ymax * scale_factor)
# plt.plot(xaxis,yaxis)
# plt.show()
