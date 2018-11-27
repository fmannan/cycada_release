import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
from imageio import imsave
import numpy as np


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)

    if not os.path.exists(opt.outdir):
        os.makedirs(opt.outdir)
    # create website
    #web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    #webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if i % 100 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
        #save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        #_, fname = os.path.split(img_path[0])
        #print(fname)
        outfilename = img_path[0].replace('testA', 'output')
        base, fname = os.path.split(outfilename)
        if not os.path.exists(base):
            os.makedirs(base)
        im_out = visuals['fake_B'].data.cpu().numpy()
        im_out = im_out.squeeze().transpose(1, 2, 0)
        im_out = np.uint8((im_out + 1) / 2 * 255)
        # print(im_out.min(), im_out.max())
        imsave(outfilename, im_out)

    #webpage.save()
