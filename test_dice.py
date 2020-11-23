import time
from options.train_options import TrainOptions
from dataloaders import CreateDataLoader, CreateValDataLoader
from models import create_model
from utils.visualizer import Visualizer
from utils.evaluation_metric import AverageMeter


if __name__ == '__main__':
    opt = TrainOptions().parse()
    opt.isTrain = False
    val_data_loader = CreateValDataLoader(opt)
    val_dataset = val_data_loader.load_data()
    val_dataset_size = len(val_data_loader)
    print('#validation images = %d' % val_dataset_size)

    model = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        visualizer.reset()

        # test
        top_1_meter = AverageMeter()
        top_1_meter1 = AverageMeter()
        top_1_meter2 = AverageMeter()
        top_1_meter3 = AverageMeter()
        top_1_meter4 = AverageMeter()
        top_1_meter5 = AverageMeter()
        top_1_meter6 = AverageMeter()
        top_1_meter7 = AverageMeter()

        for i, data in enumerate(val_dataset):
            A_path = str(data["A_paths"][0])
            B_path = str(data["B_paths"][0])

            model.set_input(data)

            dice, dsc = model.test()

            top_1_meter.update(dice)
            top_1_meter1.update(float(dsc[0]))
            top_1_meter2.update(float(dsc[1]))
            top_1_meter3.update(float(dsc[2]))
            top_1_meter4.update(float(dsc[3]))
            top_1_meter5.update(float(dsc[4]))
            top_1_meter6.update(float(dsc[5]))
            top_1_meter7.update(float(dsc[6]))

            metrics = model.get_current_metrics()

            # if opt.display_id > 0:
            #     visualizer.plot_current_dice(epoch, 1, opt, metrics)

            save_result = total_steps % opt.update_html_freq == 0
            # visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        print('Final Results:', top_1_meter.avg,
              top_1_meter1.avg,
              top_1_meter2.avg,
              top_1_meter3.avg,
              top_1_meter4.avg,
              top_1_meter5.avg,
              top_1_meter6.avg,
              top_1_meter7.avg)
        break
