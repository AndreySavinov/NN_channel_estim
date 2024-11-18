from itertools import cycle
import gc
import os,sys
import argparse
import json
import torch
from torch.optim import Adam
from tqdm import tqdm
import pandas as pd
from collections import OrderedDict
import numpy as np

# our own functions
module = sys.modules[__name__]
MODULE_ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(MODULE_ROOT_PATH+os.sep+'..')

from helpers import Scenario
import simple_algo_detector as sad_module
from simple_algo_detector import batches_learning
from helpers import cut_to_parts
from simple_algo_detector import CE_TTI


## use tensorboard writer
USE_WRITER=False
SKIP_VARIABLES_SET = set(['h_f_noisy_re','h_f_noisy_im'])

SCALE_FACTOR = 1e4


if USE_WRITER:
    from tensorboardX import SummaryWriter
    sad_module.SummaryWriter=SummaryWriter
    sad_module.USE_WRITER=True


def set_config_to_variables(scenario,config):
    global module
    for k in config:
        if k in scenario._fields:
            scenario = scenario._replace(**dict([(k,config[k]),]))
            print('scenario:' , k, '=', config[k])

        elif k == 'ml_coef':
            ml_coef_tensored = _make_tensored_ml_coef(config['ml_coef'])
            for c in module.ml_coef:
                if c not in ml_coef_tensored:
                    ml_coef_tensored[c] = module.ml_coef

            setattr(module, 'ml_coef', ml_coef_tensored)
            print('ml_coef: ',ml_coef_tensored)
        elif k in SKIP_VARIABLES_SET:
            continue
        else:
            # setting variable described in config in module namespace
            setattr(module, k, config[k])
            print(k,'=',config[k])
            
    return scenario

def _make_tensored_ml_coef(ml_coef_non_tensored):
    ml_coef_tensored = dict()
    for ml_k, ml_val in ml_coef_non_tensored.items():
        ml_coef_tensored[ml_k] = torch.tensor(ml_val, requires_grad=True)
    return ml_coef_tensored

def _get_keys_for_dict(in_dict):
    return sorted(in_dict.keys())

def get_header_for_dict(in_dict,sep=','):
    return sep.join([str(k) for k in _get_keys_for_dict(in_dict)])

def get_values_for_dict(in_dict):
    keys = _get_keys_for_dict(in_dict)
    return [in_dict[v] for v in keys]

def run_learning(config=None,scenario=None,ml_coef=None,device=None):
    """
    Function to start learning
    
    :param config: learning settings 
    :type config: dict
    """
    # some default constants
    if scenario is None:
        scenario = Scenario(SNR=30, seed=0,  RB_num=4, index=1,
                            N_TTI=1, UE_indx=0, UE_number=1, N_seeds=3, N_scenarios=40, RB_size=12, N_pilot=1, Nrx=64,
                            upsample_factor=1, Nfft=512, N_shift=100, N_response=int(448*512/2048), comb=0)
    max_iter = 10 # сколько раз вычитается пик
    
    # unpack all config here
    scenario = set_config_to_variables(scenario,config)
    sad_module.ml_version = module.ml_version

    train_files = [module.train_ONE_PILOT_DATA_DIR+os.sep+f for f in sorted(os.listdir(module.train_ONE_PILOT_DATA_DIR))]
    
    print('total train files: ',len(train_files))

    optimizer = Adam(list(module.ml_coef.values()),lr = module.Adam_lr)
    #optimizer.add_param_group(module.ml_init)

    all_ml_steps = list()

    # if USE_WRITER:
    #     sad_module.writer = SummaryWriter()

    for i in tqdm(range(1,module.epochs+1)):
        for minibatch_index,minibatch in tqdm(enumerate(cut_to_parts(train_files,batch_size=module.minibatch_size))):

            optimizer.zero_grad()
            loss,SNRs_errors = batches_learning(scenario, 
                                    module.ml_coef,
                                    max_iter, 
                                    module.SNR_range,
                                    minibatch,
                                    module.SNR_error_weights,
                                    return_SNR_batch_errors=True,
                                    device=device,n_pilots=n_pilots,dtype=torch.float32)
            
            loss = loss/(len(minibatch)*scenario.N_seeds)
            loss.backward()
            optimizer.step()
            
            detached_arr = [v.cpu().data.detach().numpy() for v in get_values_for_dict(module.ml_coef)]
            all_ml_steps.append(detached_arr)
            del detached_arr

            if not os.path.isfile(output):
                header = 'epoch,minibatch index,'+get_header_for_dict(module.ml_coef,sep=',')+\
                             ',mean minibatch total loss,'+','.join(['mean error at '+str(snr)+' SNR' for snr in module.SNR_range])
                with open(output, 'a+') as f:
                    f.write(header+'\n')

            with open(output,'a+') as f:
                f.write(str(i)+','+str(minibatch_index)+','+','.join([str(v) for v in all_ml_steps[-1]])+','+str(loss.cpu().data.detach().numpy()/len(minibatch)))
                
                # errors for each snr in train while learning
                f.write(','+','.join([str(c/len(minibatch)) for c in SNRs_errors]))
                   
                f.write('\n')

    # if USE_WRITER:
    #     sad_module.writer.export_scalars_to_json("./tensorboard/all_scalars.json")
    #     sad_module.writer.close()


def run_testing(config=None,scenario=None,ml_coef={},device=None):
    
    # some default constants
    max_iter = 10 # сколько раз вычитается пик
    
    # unpack all config here
    scenario = set_config_to_variables(scenario,config)
    sad_module.ml_version = module.ml_version

    train_files = [test_ONE_PILOT_DATA_DIR+os.sep+f for f in sorted(os.listdir(test_ONE_PILOT_DATA_DIR))]
    
    print('Starting testing with parameters (SNR would be shanged later):',scenario)
                                              
    with torch.no_grad():
        results_errors = pd.DataFrame()
        
        for test_SNR in tqdm(SNR_range):
                # setting SNR
            scenario = scenario._replace(SNR = test_SNR)

            results_errors['SNR '+str(test_SNR)+
                          ' N_seeds '+str(scenario.N_seeds)] = sad_module.test(scenario,
                                                                             ml_coef,
                                                                             max_iter, 
                                                                             train_files,use_tqdm=True,
                                                                         device=device,
                                                                         n_pilots=n_pilots,dtype=torch.float32)

        results_errors.index = [i for i in range(1,len(train_files)+1)]
        results_errors.index.name = 'scenario'

        results_errors.to_csv(output)

def jsonize_scenario_and_data(scen,h_f_noisy,ml,max_iter,device,dtype):
    json_to_call = dict()

    # filling json with scenario parameters
    for k in scen._fields:
        v = getattr(scen,k)
        json_to_call[k] = v

    # filling json with ml coeffs
    for k, v in ml.items():
        json_to_call[k] = v.data.cpu().numpy().tolist()

    # filling json with data
    assert h_f_noisy.shape[-1] == 2

    json_to_call['h_f_noisy_re'] = h_f_noisy[:,:,0].data.cpu().numpy().tolist()
    json_to_call['h_f_noisy_im'] = h_f_noisy[:,:,1].data.cpu().numpy().tolist()

    # filling json with max_iter
    json_to_call['max_iter'] = max_iter

    # special flag for scaling handling
    json_to_call['_python_call'] = True

    return json_to_call


def parse_result_json(result_json):
    h_f_recovered_pilots = torch.Tensor(result_json['h_f_recovered_pilots'])
    h_f_recovered_data = torch.Tensor(result_json['h_f_recovered_data'])
    return h_f_recovered_pilots, h_f_recovered_data


def client_CE_TTI(scen, h_f_noisy, ml, max_iter, device=None, dtype=torch.float32):
    json_to_call = jsonize_scenario_and_data(scen,h_f_noisy, ml, max_iter, device, dtype)

    import requests
    API_ENDPOINT = sad_module.host+":"+str(sad_module.port)+'/CE_TTI'

    result_json = requests.post(url=API_ENDPOINT, json=json_to_call).json()
    return parse_result_json(result_json)


def run_client_testing(config=None, scenario=None, ml_coef={}, device=None):
    # some default constants
    max_iter = 10  # сколько раз вычитается пик

    # unpack all config here
    scenario = set_config_to_variables(scenario, config)
    sad_module.ml_version = module.ml_version

    train_files = [test_ONE_PILOT_DATA_DIR + os.sep + f for f in sorted(os.listdir(test_ONE_PILOT_DATA_DIR))]

    print('Starting testing with parameters (SNR would be shanged later):', scenario)

    with torch.no_grad():
        results_errors = pd.DataFrame()

        for test_SNR in tqdm(SNR_range):
            # setting SNR
            scenario = scenario._replace(SNR=test_SNR)

            results_errors['SNR ' + str(test_SNR) +
                           ' N_seeds ' + str(scenario.N_seeds)] = sad_module.test(scenario,
                                                                       ml_coef,
                                                                       max_iter,
                                                                       train_files, use_tqdm=True,
                                                                       device=device,
                                                                       n_pilots=n_pilots, dtype=torch.float32)

        results_errors.index = [i for i in range(1, len(train_files) + 1)]
        results_errors.index.name = 'scenario'

        results_errors.to_csv(output)

def reconstruct_h_f_noisy_from_json(req_json,scenario,astype=None):
    scaleFactor = SCALE_FACTOR
    is_python_call=False

    if '_python_call' in req_json:
        if req_json['_python_call']:
            scaleFactor = 1
            is_python_call = True

    h_f_re_json = req_json['h_f_noisy_re']
    h_f_im_json = req_json['h_f_noisy_im']

    h_f_re = np.array(h_f_re_json)
    h_f_im = np.array(h_f_im_json)

    N_used = scenario.RB_num * scenario.RB_size

    # two pilots in input
    if h_f_re.shape[-1]==2:
        #h_data = torch.zeros(scenario.Nrx, N_used, N_data_sym, 2, dtype=astype)
        h_pilot = torch.zeros(scenario.Nrx, N_used, 2,2, device=device, dtype=astype)
        h_pilot[:,:,0,0] = torch.Tensor(h_f_re[:,:,0])*scaleFactor
        h_pilot[:,:,1,0] = torch.Tensor(h_f_re[:,:,1])*scaleFactor
        h_pilot[:,:,0,1] = torch.Tensor(h_f_im[:,:,0])*scaleFactor
        h_pilot[:,:,1,1] = torch.Tensor(h_f_im[:,:,1])*scaleFactor

    else:
        h_pilot = torch.zeros(scenario.Nrx, N_used, 2, device=device, dtype=astype)
        h_pilot[:,:,0] = torch.Tensor(h_f_re)*scaleFactor
        h_pilot[:,:,1] = torch.Tensor(h_f_im)*scaleFactor

    return h_pilot,is_python_call

def construct_json_from_recovered(h_f_recovered_pilots, h_f_recovered_data,is_python_call=False):
    result_json = dict()

    if is_python_call:
        # call from python, acc to current code? h_f_recovered_data and h_f_recovered_pilots can have different shape
        result_json['h_f_recovered_pilots'] = h_f_recovered_pilots.data.cpu().numpy().tolist()
        result_json['h_f_recovered_data'] = h_f_recovered_data.data.cpu().numpy().tolist()
    else:
        scale_back_factor = 1/SCALE_FACTOR
        # call from matlab, h_f_recovered_data and h_f_recovered_pilots must be the same
        rec__pilots_numpy = h_f_recovered_pilots.data.cpu().numpy()*scale_back_factor

        result_json['h_f_recovered_pilots'] = np.swapaxes(np.swapaxes(np.array([rec__pilots_numpy,rec__pilots_numpy]),0,-2),0,1).tolist()
        result_json['h_f_recovered_data'] = (h_f_recovered_data.data.cpu().numpy()*scale_back_factor).tolist()

    return result_json


def run_serve(config=None,scenario=None,ml_coef={}, device=None):
    # unpack all config here
    scenario = set_config_to_variables(scenario, config)
    sad_module.ml_version = module.ml_version

    with torch.no_grad():
        from aiohttp import web

        async def handle(request):
            name = request.match_info.get('name', "Anonymous")
            req_json = await request.json()

            # set scenario parametersfrom json request
            global global_scenario
            scenario = set_config_to_variables(global_scenario, req_json)

            # reconstruct torch h_f_noisy matrices from json
            h_f_noisy,is_python_call = reconstruct_h_f_noisy_from_json(req_json,scenario)

            # call CE_TTI
            h_f_recovered_pilots, h_f_recovered_data = CE_TTI(scenario, h_f_noisy, ml_coef, module.max_iter)

            # put results to json and return it back
            recovered_json_data = construct_json_from_recovered(h_f_recovered_pilots, h_f_recovered_data,is_python_call)

            return web.json_response(recovered_json_data)

        async def help(request):
            text = str(open('./Readme.md').readlines())
            return web.Response(text=text)

        app = web.Application(client_max_size=1024*1024*10)
        app.add_routes([web.get('/', help),
                        web.post('/CE_TTI', handle)])

        web.run_app(app,port=config['port'],host=config['host'])


def _get_default_ml_coeffs():
    d = {"softm_main_scale": 2.1,
                     "softm_power": 1.5,
                     "softm_power_diff": 0.0,
                     "softm_SNR_diff": 0.0,
                     "softm_SNR": -0.3,
                     "sigm_power": 0.0,
                     "sigm_power_diff": 0.0,
                     "sigm_SNR": -0.3,
                     "sigm_SNR_diff": 0.0,
                     "bayes_c0": 0.04,
                     "bayes_c1": 0.7}
    return json.dumps(d).replace(' ','')
        
def load_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='My example explanation')
        
    parser.add_argument('-c',
        '--config',
        type=str,
        default=MODULE_ROOT_PATH+os.sep+'/config/config.json',
        help='filepath to config json file')
    
    parser.add_argument('-o',
        '--output',
        type=str,
        default='output.log',
        help='filepath to store experiments results')

    parser.add_argument('--ml_coef',
                        type=str,
                        default='dict()',
                        help='starting ml_init for algorithm')

    parser.add_argument('--host',
                        type=str,
                        default='0.0.0.0',
                        help='host for serving CE_TTI')

    parser.add_argument('--port',
                        type=str,
                        default='8080',
                        help='port for serving CE_TTI')
    
    parser.add_argument("--train", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="use algo_runner for model train")
                        
    parser.add_argument("--test", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="use algo_runner for model test")

    parser.add_argument("--client", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="use algo_runner for model test as client connecting to other server")

    parser.add_argument("--serve", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="use algo_runner for serving model CE_TTI function as service")
                        
    args = parser.parse_args()
    if args.client:
        args.test=True
                        
    if (int(args.train) + int(args.test)+int(args.serve))>1:
        parser.error('Can`t do more than one of --train,--test,--serve at once!')
                        
    if (int(args.train) + int(args.test)+int(args.serve))==0:
        parser.error('Please specify --test or --train action')

    if not os.path.isfile(args.config):
        parser.error('Config file "'+args.config+'" does not exist!"')

    if not args.serve:
        if os.path.isfile(args.output):
            parser.error('Output file "'+args.output+'" already exists!"')

    return args

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('torch device is:',device)

    # setting default type
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    
    # load args
    args = load_args()
    ml_coef_dict = eval(args.ml_coef)

    config_path = args.config
    
    config = json.loads(open(config_path).read())

    ## replace config ml_coef with overrided ml_coef
    for c in config['ml_coef']:
        if c not in ml_coef_dict:
            ml_coef_dict[c] = config['ml_coef'][c]
            
    ml_coef = _make_tensored_ml_coef(ml_coef_dict)

    if ('host' not in config) or (config['host'] != args.host):
        config['host'] = args.host

    if ('port' not in config) or (config['port'] != int(args.port)):
        config['port'] = int(args.port)

    for k in config:
        if k=='SNR_error_weights':
            config[k] = dict([(float(snr),weight) for snr,weight in config[k].items()])
            
    config['output'] = args.output
    
    scenario = Scenario(SNR=30, seed=0,  RB_num=4, index=1,
                            N_TTI=1, UE_indx=0, UE_number=1, N_seeds=3, N_scenarios=40, RB_size=12, N_pilot=1, Nrx=64,
                            upsample_factor=1, Nfft=512, N_shift=100, N_response=int(448*512/2048), comb=0)

    module.ml_version=5


    global global_scenario
    global_scenario = scenario

    if args.train:  
        run_learning(config=config,scenario=scenario,ml_coef=ml_coef,device=device)
    elif args.test:
        if args.client:
            sad_module.host = 'http://' + config['host']
            sad_module.port = config['port']

            sad_module.CE_TTI = client_CE_TTI

            run_client_testing(config=config, scenario=scenario, ml_coef=ml_coef, device=device)
        else:
            run_testing(config=config,scenario=scenario,ml_coef=ml_coef,device=device)
    else:
        # serve aiohttp server with model
        run_serve(config=config,scenario=scenario,ml_coef=ml_coef,device=device)
