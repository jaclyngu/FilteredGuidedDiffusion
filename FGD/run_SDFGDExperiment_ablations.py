from apy import *
from apy.amedia import *
import torchvision.transforms as transforms
import torch as th
from fgdex import *
# import sys
sys.path.append(os.path.abspath('../'))
import StableDiffusion
import Text2Im
from apy.utils import *
import Img2Img
import time
from SDFGDExperiment import *
# import os
# import numbers
import numpy as np

invert_portion=1.
output_dirname='../outputs_param_search_FGD_stdAblations'#_DDIMInvert'+str(invert_portion)
#dataset dir
parent_dir = '../dataset/param_search/'
styles = open(parent_dir+'styles.txt').readline().split(', ')

spatial_stds=[3, 5, 11]
value_stds=[0.1, 0.35, 0.5]
t_ends=[10]
t_starts=[50]#[int(50*invert_portion)]
normalize_guidances=[1]
d_structures=[0.05, 0.2, 0.5]
use_xT = True
plms = True
SDEdit_strengths = [-1]
os.makedirs(output_dirname, exist_ok=True)

for f in sorted(os.listdir(parent_dir))[-7:]:
    data_dir = parent_dir + f
    if not os.path.isdir(data_dir):
        print(data_dir, 'is not a data folder!')
        continue
    for img_f in os.listdir(data_dir):
        if ('jpg' in img_f) or ('png' in img_f) or ('jpeg' in img_f):
            guide_im = GuideImage(os.path.join(data_dir, img_f))

    try:
        objects = open(os.path.join(data_dir, 'untitled.txt')).readline().split(', ')
    except:
        print('no untitled.txt!')
        continue

    if use_xT:
        if invert_portion != 1.:
            name = str(invert_portion)+'_z_enc.pt'
        else:
            name = 'z_enc.pt'
            
        if name in os.listdir(data_dir):
            xT_latent = torch.load(os.path.join(data_dir, name))
        else:
            xT_latent = os.path.join(data_dir, name)
    else:
        xT_latent = None
        
    for style in styles:
        for obj in objects:
            prompt = style +' '+ obj
            prompt_dir = os.path.join(output_dirname, f, prompt)
            for spatial_std in spatial_stds:
                for value_std in value_stds:
                    if spatial_std==5 and value_std==0.35:
                        continue
                    bilat_dir = os.path.join(prompt_dir, 'b'+'_'.join([str(spatial_std), str(value_std)]))
                    for t_start in t_starts:
                        for t_end in t_ends:
                            if t_start - t_end > 0:
                                time_dir = os.path.join(bilat_dir, 't'+'_'.join([str(t_start), str(t_end)]))
                                for d_structure in d_structures:
                                    structure_dir = os.path.join(time_dir, 's'+str(d_structure))

                                    for normalize_guidance in normalize_guidances:
                                        norm_dir = os.path.join(structure_dir, 'n'+str(normalize_guidance))
                                        for strength in SDEdit_strengths:
                                            output_dir = os.path.join(norm_dir, 'sde'+str(strength))
                                            if os.path.exists(norm_dir) and (output_dir in os.listdir(norm_dir)):
                                                user_decision = input(output_dirname+output_dir+' already exists, overwrite? [Y/N]')
                                                if user_decision != 'Y':
                                                    print("No overwrite, aborting...")
                                                    sys.exit()
                                            print(output_dir)
                                            os.makedirs(output_dir, exist_ok=True)
                                            start = time.time()
                                            experiment = SDFGDExperiment.Create(
                                                directory = output_dir,
                                                plms=plms,
                                                guide_image=guide_im,
                                                guide_text=prompt,
                                                target_detail=d_structure,
                                                t_end=t_end,
                                                sigmas=[spatial_std, spatial_std, value_std],
                                                t_total=50,
                                                t_start=t_start,
                                                normalize_guidance = normalize_guidance,
                                                guidance_exponent=1.0,
                                                seed=8,
                                                xT_latent=xT_latent,
                                                invert_portion=invert_portion
                                                )
                                            exp = experiment.Evaluate()
                                            entire_time = time.time()-start
                                            print('time', entire_time, experiment.exe_time)
                                            with open(os.path.join(output_dir, 'entire_time.txt'), 'w') as f1:
                                                f1.write(str(entire_time))
                                            with open(os.path.join(output_dir, 'exe_time.txt'), 'w') as f2:
                                                f2.write(str(experiment.exe_time))
                                            resultim = Image(pixels=experiment.grid_results[0])
                                            experiment.WriteResults(output_dir)
                                            resultim.writeToFile(os.path.join(output_dir, 'generated.png'))


                                
