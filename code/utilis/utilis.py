from tensorflow.python.client import device_lib

def get_available_gpus():
    """ call it to get info on names of avialbale GPUs """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']