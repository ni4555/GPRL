import pickle
import matplotlib.pyplot as plt

#result_f = open('./result/reward_save/767999_reward_GP.pkl', 'rb')
result_f = open('./result/reward_save/reward_GP.pkl', 'rb')
reward_data = pickle.load(result_f)
extracted_reward = reward_data[0]
print(extracted_reward)


reward_per_iter = []
cnt = 0
average_reward = 0
for reward in extracted_reward:
    average_reward += reward
    cnt += 1
    if cnt == 512*1:
        reward_per_iter.append(average_reward / 512*1)
        cnt = 0
        average_reward = 0
plt.plot(reward_per_iter)
plt.show()
