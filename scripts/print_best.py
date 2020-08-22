import os
import pandas as pd

checkpoint_path = 'checkpoints'

for network_name in os.listdir(checkpoint_path):
    network_path = os.path.join(checkpoint_path, network_name)
    
    training_log = pd.read_csv(os.path.join(network_path, network_name + '_log.txt'))
    eval_log = pd.DataFrame.copy(training_log[training_log['mode']=='Eval']).reset_index(drop=True)
    best_epoch = eval_log['loss'].idxmin()
    # print('The best network was', network_name + '_{0:03d}.pt'.format(best_epoch), 'with a loss of ', eval_log['loss'].min())
    print(os.path.join(network_path, network_name + '_{0:03d}.pt'.format(best_epoch)))
    
    # TODO: Automatic changing of the name of the network, so that we can use the sampling script.