from nbeats_pytorch.model import NBeatsNet
import nbeats_additional_functions as naf
import os
import torch
from torch import optim

checkpoint_name_BASE = "nbeats_checkpoint.th"
checkpoint_training_base = "_training.th"

data_dir = os.getcwd() + "/data/"

d = [x[0] for x in os.walk(data_dir)]
dirs = []
for directory_name in d:
    if ',' not in directory_name:
        dirs.append(directory_name)
dirs = dirs[1:]

threshold = 0.0001
limit = 100
plot_eval = False

#Bart

#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cuda')
torch.set_default_tensor_type('torch.cuda.FloatTensor')
print(device)

#input = input.cuda()
#target = target.cuda() 

forecast_length = 500
backcast_length = 3 * forecast_length
batch_size = 256

for folder_name in dirs:
    name = folder_name.split("/")[-1]
    checkpoint_name = name + "_" + checkpoint_name_BASE
    training_checkpoint = name + checkpoint_training_base
    print(name)

    if os.path.isfile(checkpoint_name):
        continue
    print("PRZED SIECIA")

    net = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
                    forecast_length=forecast_length,
                    thetas_dims=[7, 8],
                    nb_blocks_per_stack=3,
                    backcast_length=backcast_length,
                    hidden_layer_units=128,
                    share_weights_in_stack=False,
                    device=device)
    optimiser = optim.Adam(net.parameters())
    
    test_losses = []
    old_eval = 100
    the_lowest_error = [100]    

    actual_class_dir = data_dir  + name + "/"
    print(actual_class_dir)
    
    for (_, dirs, files) in os.walk(actual_class_dir):
        iteration = 0
        difference = 100 #BART -> if the training takes forever pls move this line out of this loop
        for file in files:
            print(file)
            i = 0
            

            if 'mat' in file:
                continue

            iteration += 1
            if iteration > 30 or difference < threshold:
                break

            data, x_train, y_train, x_test, y_test, norm_constant = naf.one_file_training_data(actual_class_dir, file, forecast_length, backcast_length, batch_size)
          
            while difference > threshold and i < limit  :
                i += 1
                global_step = naf.train_full_grad_steps(data, device, net, optimiser, test_losses, training_checkpoint, x_train.shape[0])
                new_eval = naf.evaluate_training(backcast_length, forecast_length, net, norm_constant, test_losses, x_test, y_test, the_lowest_error, device)
                print(f"GlobalStep: {global_step}, New evaluation sccore: {new_eval}")
                if new_eval < old_eval:
                    difference = old_eval - new_eval
                    old_eval = new_eval
                    with torch.no_grad():
                        print("New evaluation value:", new_eval, "  iteration:", i)
                        print("Saving...")
                        new_checkpoint_name =  str(checkpoint_name[:-3]+str(len(test_losses))+".th")
                        naf.save(new_checkpoint_name, net, optimiser, global_step )

                
