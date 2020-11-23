import time
from options.train_options import TrainOptions
from dataloaders import CreateDataLoader, CreateValDataLoader
from models import create_model
from utils.visualizer import Visualizer
from utils.evaluation_metric import AverageMeter


if __name__ == '__main__':
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('# The number of training images = %d' % dataset_size)

    val_data_loader = CreateValDataLoader(opt)
    val_dataset = val_data_loader.load_data()
    val_dataset_size = len(val_data_loader)
    print('# Validation images = %d' % val_dataset_size)

    model = create_model(opt)
    visualizer = Visualizer(opt)
    total_iters = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()

        # train
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ optimize ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            model.optimize_parameters()

            # # ~~~~~~~~~~~~~~~~~~~~~~~~ Use Visdom for Visualization ~~~~~~~~~~~~~~~~~~~~~~~~
            # if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
            #     save_result = total_iters % opt.update_html_freq == 0
            #     # model.compute_visuals()
            #     visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        model.update_learning_rate(opt, float(model.get_current_losses()["total"]))
