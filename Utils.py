import matplotlib.pyplot as plt
import os
import shutil

def plot_avg_reward(train_id, avg_reward_list):
    plt.plot(avg_reward_list)
    plt.xlabel("episode")
    plt.ylabel("avg reward")
    # save fig
    fig_name = str(train_id) + '.png'
    fig_path = 'Log/' + fig_name
    plt.savefig(fig_path)
    plt.show()

# clear previous saved outputs
def clear_history(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))