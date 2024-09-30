
import os 
import glob 
import torch 

def delete_last_model(model_dir, symbol):

    last_model = glob.glob(f"{model_dir}/{symbol}*.pt")
    if len(last_model) != 0:
        os.remove(last_model[0])


def save_new_model_and_delete_last(model, save_path, delete_symbol=None):
    save_dir = os.path.dirname(save_path)

    os.makedirs(save_dir, exist_ok=True)
    if delete_last_model is not None:
        delete_last_model(save_dir, delete_symbol)
    
    torch.save(model.state_dict(), save_path)

    print(f"model is saved in {save_path}")


def load_model_and_resume_training(model, load_path, optimizer=None):
    if not os.path.isfile(load_path):
        raise FileNotFoundError(f"No model found at {load_path}")

    model.load_state_dict(torch.load(load_path))
    print(f"Model loaded from {load_path}")

    if optimizer:
        optimizer_state_path = load_path.replace('.pt', '_optimizer.pt')
        if os.path.isfile(optimizer_state_path):
            optimizer.load_state_dict(torch.load(optimizer_state_path))
            print(f"Optimizer state loaded from {optimizer_state_path}")
        else:
            print(f"No optimizer state found at {optimizer_state_path}")

    return model, optimizer