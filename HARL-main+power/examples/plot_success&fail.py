import numpy as np
from matplotlib import pyplot as plt, ticker

success_hist = []
fail_collision_hist=[]
fail_pu_hist=[]
reward=[]
# file_folder = './myresult/happo32-8-3-4-xiaorong/'
file_folder = './myresult/happo-32-84-xiao/'

label_list = ['HAPPO-DSA', 'MAPPO-DSA', 'HAPPO-DSA(share param)', 'HAPPO-DSA(fixed order)','HAA2C-DSA',]
marker_list = ['r-*', 'c-^', 'm-o', 'g-d', 'b-x', 'b-^']

n_curve=5
n_channel=32
for n in range(n_curve):
    success_hist.append(np.load(file_folder + '/success_hist_%d.npy' % (n + 1)))
    fail_collision_hist.append(np.load(file_folder + '/fail_collision_hist_%d.npy' % (n + 1)))
    fail_pu_hist.append(np.load(file_folder + '/fail_PU_hist_%d.npy' % (n + 1)))
    reward.append(np.load(file_folder + '/average_reward_%d.npy' % (n + 1)))

if n_channel == 6:
    mean_average_time = 4
elif n_channel == 22:
    mean_average_time = 3
elif n_channel == 32:
    mean_average_time = 15
elif n_channel == 18:
    mean_average_time =15#18

record_number = int(success_hist[0].size / mean_average_time)

success_hist_mean = []
fail_collision_hist_mean = []
fail_pu_hist_mean=[]
reward_mean=[]
batch_size=2000
for n in range(n_curve):
    success_hist_mean.append(np.zeros(record_number))
    fail_collision_hist_mean.append(np.zeros(record_number))
    fail_pu_hist_mean.append(np.zeros(record_number))
    reward_mean.append(np.zeros(record_number))
    for k in range(record_number):
        index = np.arange(k * mean_average_time, (k + 1) * mean_average_time)
        success_hist_mean[n][k] = np.mean(success_hist[n][index])
        fail_collision_hist_mean[n][k] = np.mean(fail_collision_hist[n][index])
        fail_pu_hist_mean[n][k] = np.mean(fail_pu_hist[n][index])
        reward_mean[n][k]=np.mean(reward[n][index])

total_record_number = (np.arange(record_number) + 1) * batch_size * mean_average_time*0.5
plt.figure()
    #plt.plot(total_record_number, success_hist_mean[0] / batch_size, marker_list[0], label=label_list[0])
fig, ax = plt.subplots()
for i in range(n_curve):
    ax.plot(total_record_number, success_hist_mean[i], marker_list[i], label=label_list[i])

ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
# ax.legend(loc='upper right')
plt.legend(loc='lower right')
#plt.legend(loc='lower right')
if n_channel == 6:
    plt.ylim(0.3, 0.9)
elif n_channel == 22:
    plt.ylim(0.7, 1)
elif n_channel == 8:
    plt.ylim(0.1, 1.0)
elif n_channel == 18:
    plt.ylim(0.5, 1.08)
elif n_channel == 32:
    plt.ylim(0.5, 1.08)
plt.legend(loc='lower right')
font = {'family': 'Times New Roman', 'size': 12}

plt.ylabel('Average success rate', fontdict=font)
plt.xlabel('Training steps', fontdict=font)
fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
plt.figure()
    #plt.plot(total_record_number, success_hist_mean[0] / batch_size, marker_list[0], label=label_list[0])
fig, ax = plt.subplots()
# plt.plot(total_record_number, success_hist_mean[0] / batch_size, marker_list[0], label=label_list[0])
for i in range(n_curve):
    ax.plot(total_record_number, fail_collision_hist_mean[i], marker_list[i], label=label_list[i])

ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
plt.legend(loc='upper right')
if n_channel == 6:
    plt.ylim(0.3, 0.9)
elif n_channel == 22:
    plt.ylim(0.7, 1)
elif n_channel == 8:
    plt.ylim(0.1, 1.0)
elif n_channel == 18:
    plt.ylim(0.0, 0.5)
elif n_channel == 32:
    plt.ylim(0.0, 0.5)
    # plt.ylim(0.3, 1)
plt.legend(loc='upper right')
font = {'family': 'Times New Roman', 'size': 12}

plt.ylabel('Average collision (with SU) rate', fontdict=font)
plt.xlabel('Training steps', fontdict=font)
fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)

plt.figure()

fig, ax = plt.subplots()
for i in range(n_curve):
    ax.plot(total_record_number, reward_mean[i], marker_list[i], label=label_list[i])

# ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
plt.legend(loc='upper right')
if n_channel == 6:
    plt.ylim(0.3, 0.9)
elif n_channel == 22:
    plt.ylim(0.7, 1)
elif n_channel == 8:
    plt.ylim(0.1, 1.0)
elif n_channel == 18:
    plt.ylim(2.0, 8)
elif n_channel == 32:
    plt.ylim(2.0, 32)
    # plt.ylim(0.3, 1)
plt.legend(loc='lower right')
font = {'family': 'Times New Roman', 'size': 12}

plt.ylabel('Average reward', fontdict=font)
plt.xlabel('Training steps', fontdict=font)
fig.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.15)

plt.figure()



plt.show()

