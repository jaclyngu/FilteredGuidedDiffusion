from apy import *
from apy.amedia import *
import torchvision.transforms as transforms
import torch as th
from fgdex import *
# import sys
# sys.path.append(os.path.abspath('../'))
# import StableDiffusion
import Text2Im
from apy.utils import *
import Img2Img
import time
from SDFGDExperiment import *
# import os
# import numbers
import numpy as np


output_dirname='../outputs_Landscape_subset_PnP+FGD'
#dataset dir
parent_dir = '../dataset/Landscape_subset/'
styles = open(parent_dir+'styles.txt').readline().split(', ')

spatial_std=5
value_std=0.35
t_ends=[10, 25]
t_starts=[50]
normalize_guidances=[3, 1, 0]
d_structures=[0.05, 0.2, 0.5]
use_xT = True
# plms = False
SDEdit_strengths = [-1]
os.makedirs(output_dirname, exist_ok=True)

for f in sorted(os.listdir(parent_dir)):
    if f in os.listdir(output_dirname):
        user_decision = input(output_dirname+f+' already exists, overwrite? [Y/N]')
        if user_decision != 'Y':
            print("No overwrite, aborting...")
            sys.exit()

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
        xT_latent = torch.load(os.path.join(data_dir, 'z_enc.pt'))
    else:
        xT_latent = None
        
    for style in styles:
        for obj in objects:
            prompt = style +' '+ obj
            prompt_dir = os.path.join(output_dirname, f, prompt)
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
                                    print(output_dir)
                                    os.makedirs(output_dir, exist_ok=True)
                                    start = time.time()
                                    experiment = SDFGDExperiment.Create(
                                        directory = output_dir,
#                                         plms=plms,
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
                                        )
                                    exp = experiment.EvaluatePnP()
                                    entire_time = time.time()-start
                                    print('time', entire_time, experiment.exe_time)
                                    with open(os.path.join(output_dir, 'entire_time.txt'), 'w') as f1:
                                        f1.write(str(entire_time))
                                    with open(os.path.join(output_dir, 'exe_time.txt'), 'w') as f2:
                                        f2.write(str(experiment.exe_time))
                                    resultim = Image(pixels=experiment.grid_results[0])
                                    experiment.WriteResults(output_dir)
                                    resultim.writeToFile(os.path.join(output_dir, 'generated.png'))


                                
