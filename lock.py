import argparse
import os
import random
import time
import torch as ch
from robustness.datasets import CIFAR
from robustness.model_utils import make_and_restore_model


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--orig-data', default='./data/cifar10', help='path to the original dataset')
    parser.add_argument('--enc-data', default='./data', help='path to output encrypted dataset')
    parser.add_argument('--resume-path', default='./logs/checkpoints/dir/resnet50/checkpoint.pt.best', help='path to checkpoint to resume from')
    parser.add_argument('--workers', type=int, help='data loading workers', default=8)
    parser.add_argument('--batch-size', type=int, default=128, help='batch size for data loading')
    parser.add_argument('--arch', default='resnet50', help='architecture')
    parser.add_argument('--eps', type=float, default=0.5, help='adversarial perturbation budget')
    parser.add_argument('--attack-lr', type=float, default=0.1, help='step size for PGD')
    parser.add_argument('--attack-steps', type=int, default=100, help='number of steps for adversarial attack')
    parser.add_argument('--enc-method', default='basic', choices=['basic', 'mixup', 'horiz', 'mixandcat'], help='encryption method')
    parser.add_argument('--alpha', type=float, default=0.5, help='hyperparameter in horizontal concat')
    parser.add_argument('--lambd', type=float, default=0.5, help='hyperparameter in mixup')
    parser.add_argument('--manual-seed', type=int, default=23, help='manual seed')

    opt = parser.parse_args()

    if opt.manual_seed is None:
        opt.manual_seed = random.randint(1, 10000)
    print("Random Seed: ", opt.manual_seed)
    ch.manual_seed(opt.manual_seed)

    kwargs = {
        'constraint': '2',
        'eps': opt.eps,
        'step_size': opt.attack_lr,
        'iterations': opt.attack_steps,
        'targeted': True,
        'do_tqdm': False
    }

    ds = CIFAR(opt.orig_data)
    train_loader, test_loader = ds.make_loaders(workers=opt.workers,
                                                batch_size=opt.batch_size,
                                                data_aug=False,
                                                shuffle_train=False,
                                                )

    model, _ = make_and_restore_model(arch=opt.arch, dataset=ds, resume_path=opt.resume_path)
    model.eval()


    if opt.enc_method == 'basic':
        generate_train_data(train_loader, model, kwargs, opt)
        generate_test_data(test_loader, model, kwargs, opt)

    elif opt.enc_method == 'mixup':
        generate_train_data_mixup(train_loader, model, kwargs, opt)
        generate_test_data_mixup(test_loader, model, kwargs, opt)

    elif opt.enc_method == 'horiz':
        generate_train_data_horiz(train_loader, model, kwargs, opt)
        generate_test_data_horiz(test_loader, model, kwargs, opt)

    elif opt.enc_method == 'mixandcat':
        generate_train_data_mixandcat(train_loader, model, kwargs, opt)
        generate_test_data_mixandcat(test_loader, model, kwargs, opt)


def generate_train_data(train_loader, model, kwargs, opt):
    train_image = []
    train_label = []
    for i, (im, label) in enumerate(train_loader, 0):

        time_ep = time.time()
        print("Process train batch %d" % (i))
        im = im.cuda()
        targ = map_label_target(label).cuda()

        model_prediction, im_adv = model(im, targ, make_adv=True, **kwargs)
        print("Attack success rate: %.4f" % (ch.sum(targ == ch.argmax(model_prediction, dim=1)).item() / len(im)))

        train_image.append(im_adv.cpu())
        train_label.append(label.cpu())
        time_ep = time.time() - time_ep
        print("Time:%.4f" % (time_ep))

    ch.save(train_image, os.path.join(opt.enc_data, 'train_image_basic'))
    ch.save(train_label, os.path.join(opt.enc_data, 'train_label_basic'))
    print("Save train set finished")


def generate_test_data(test_loader, model, kwargs, opt):

    test_image = []
    test_label = []
    correct = 0
    for i, (im, label) in enumerate(test_loader, 0):

        time_ep = time.time()
        print("Process testset batch %d" % (i))
        im = im.cuda()
        predict, _ = model(im)
        label_pred = ch.argmax(predict, dim=1)
        targ = map_label_target(label_pred)
        correct += ch.sum(label == label_pred.cpu()).item()

        targ = targ.cuda()
        model_prediction, im_adv = model(im, targ, make_adv=True, **kwargs)
        print("Attack success rate: %.4f" % (ch.sum(targ == ch.argmax(model_prediction, dim=1)).item() / len(im)))

        test_image.append(im_adv.cpu())
        test_label.append(label.cpu())
        time_ep = time.time() - time_ep
        print("Time:%.4f" % (time_ep))

    ch.save(test_image, os.path.join(opt.enc_data, 'test_image_basic'))
    ch.save(test_label, os.path.join(opt.enc_data, 'test_label_basic'))
    print("Save test set finished")
    print("Orig test set acc:%.4f" % (correct / 10000))


def generate_train_data_horiz(train_loader, model, kwargs, opt):
    train_image = []
    train_label = []
    for i, (im, label) in enumerate(train_loader, 0):

        time_ep = time.time()
        print("process train batch %d" % (i))
        im = im.cuda()
        targ = map_label_target(label).cuda()
        targ2 = map_label_target2(label).cuda()

        model_prediction, im_adv = model(im, targ, make_adv=True, **kwargs)
        print("Attack success rate1: %.4f" % (ch.sum(targ == ch.argmax(model_prediction, dim=1)).item() / len(im)))

        model_prediction2, im_adv2 = model(im, targ2, make_adv=True, **kwargs)
        print("Attack success rate2: %.4f" % (ch.sum(targ2 == ch.argmax(model_prediction2, dim=1)).item() / len(im)))

        # bs * 3 * height * width
        alpha = opt.alpha
        dim_W = int(alpha * im.size()[3])
        im_adv_left = im_adv[:, :, :, :dim_W]
        im_adv_right = im_adv2[:, :, :, dim_W:]
        im_adv = ch.cat((im_adv_left, im_adv_right), dim=3)

        train_image.append(im_adv.cpu())
        train_label.append(label.cpu())
        time_ep = time.time() - time_ep
        print("Time:%.4f" % (time_ep))

    ch.save(train_image, os.path.join(opt.enc_data, 'train_image_' + str(int(opt.alpha * 100)) + '_horiz'))
    ch.save(train_label, os.path.join(opt.enc_data, 'train_label_' + str(int(opt.alpha * 100)) + '_horiz'))
    print("Save train set finished")


def generate_test_data_horiz(test_loader, model, kwargs, opt):

    test_image = []
    test_label = []
    correct = 0
    for i, (im, label) in enumerate(test_loader, 0):
        time_ep = time.time()
        print("Process testset batch %d" % (i))
        im = im.cuda()
        predict, _ = model(im)
        label_pred = ch.argmax(predict, dim=1)
        targ = map_label_target(label_pred).cuda()
        targ2 = map_label_target2(label_pred).cuda()
        correct += ch.sum(label == label_pred.cpu()).item()

        model_prediction, im_adv = model(im, targ, make_adv=True, **kwargs)
        print("Attack success rate1: %.4f" % (ch.sum(targ == ch.argmax(model_prediction, dim=1)).item() / len(im)))

        model_prediction2, im_adv2 = model(im, targ2, make_adv=True, **kwargs)
        print("Attack success rate2: %.4f" % (ch.sum(targ2 == ch.argmax(model_prediction2, dim=1)).item() / len(im)))

        # bs * 3 * height * width
        alpha = opt.alpha
        dim_W = int(alpha * im.size()[3])
        im_adv_left = im_adv[:, :, :, :dim_W]
        im_adv_right = im_adv2[:, :, :, dim_W:]
        im_adv = ch.cat((im_adv_left, im_adv_right), dim=3)

        test_image.append(im_adv.cpu())
        test_label.append(label.cpu())

        time_ep = time.time() - time_ep
        print("time:%.4f" % (time_ep))

    ch.save(test_image, os.path.join(opt.enc_data, 'test_image_' + str(int(opt.alpha * 100)) + '_horiz'))
    ch.save(test_label, os.path.join(opt.enc_data, 'test_label_' + str(int(opt.alpha * 100)) + '_horiz'))
    print("Save test set finished")
    print("Orig test set acc:%.4f" % (correct / 10000))


def generate_train_data_mixup(train_loader, model, kwargs, opt):
    train_image = []
    train_label = []
    for i, (im, label) in enumerate(train_loader, 0):
        time_ep = time.time()
        print("process train batch %d" % (i))
        im = im.cuda()
        targ = map_label_target(label).cuda()
        targ2 = map_label_target2(label).cuda()

        model_prediction, im_adv = model(im, targ, make_adv=True, **kwargs)
        print("Attack success rate1: %.4f" % (ch.sum(targ == ch.argmax(model_prediction, dim=1)).item() / len(im)))

        model_prediction2, im_adv2 = model(im, targ2, make_adv=True, **kwargs)
        print("Attack success rate2: %.4f" % (ch.sum(targ2 == ch.argmax(model_prediction2, dim=1)).item() / len(im)))

        # mixup
        lambd = opt.lambd
        im_adv = lambd * im_adv + (1 - lambd) * im_adv2

        train_image.append(im_adv.cpu())
        train_label.append(label.cpu())
        time_ep = time.time() - time_ep
        print("Time:%.4f" % (time_ep))

    ch.save(train_image, os.path.join(opt.enc_data, 'train_image_' + str(int(opt.lambd * 100)) + '_mixup'))
    ch.save(train_label, os.path.join(opt.enc_data, 'train_label_' + str(int(opt.lambd * 100)) + '_mixup'))
    print("Save train set finished")


def generate_test_data_mixup(test_loader, model, kwargs, opt):

    test_image = []
    test_label = []
    correct = 0
    for i, (im, label) in enumerate(test_loader, 0):
        time_ep = time.time()
        print("Process testset batch %d" % (i))
        im = im.cuda()
        predict, _ = model(im)
        label_pred = ch.argmax(predict, dim=1)
        targ = map_label_target(label_pred).cuda()
        targ2 = map_label_target2(label_pred).cuda()
        correct += ch.sum(label == label_pred.cpu()).item()

        model_prediction, im_adv = model(im, targ, make_adv=True, **kwargs)
        print("Attack success rate1: %.4f" % (ch.sum(targ == ch.argmax(model_prediction, dim=1)).item() / len(im)))

        model_prediction2, im_adv2 = model(im, targ2, make_adv=True, **kwargs)
        print("Attack success rate2: %.4f" % (ch.sum(targ2 == ch.argmax(model_prediction2, dim=1)).item() / len(im)))

        # mixup
        lambd = opt.lambd
        im_adv = lambd * im_adv + (1 - lambd) * im_adv2

        test_image.append(im_adv.cpu())
        test_label.append(label.cpu())
        time_ep = time.time() - time_ep
        print("time:%.4f" % (time_ep))

    ch.save(test_image, os.path.join(opt.enc_data, 'test_image_' + str(int(opt.lambd * 100)) + '_mixup'))
    ch.save(test_label, os.path.join(opt.enc_data, 'test_label_' + str(int(opt.lambd * 100)) + '_mixup'))
    print("Save test set finished")
    print("Orig test set acc:%.4f" % (correct / 10000))


def generate_train_data_mixandcat(train_loader, model, kwargs, opt):
    train_image = []
    train_label = []
    for i, (im, label) in enumerate(train_loader, 0):
        time_ep = time.time()
        print("process train batch %d" % (i))
        targ = map_label_target(label).cuda()
        targ2 = map_label_target2(label).cuda()
        targ3 = map_label_target3(label).cuda()
        targ4 = map_label_target4(label).cuda()
        im = im.cuda()

        model_prediction, im_adv = model(im, targ, make_adv=True, **kwargs)
        print("Attack success rate1: %.4f" % (ch.sum(targ == ch.argmax(model_prediction, dim=1)).item() / len(im)))

        model_prediction2, im_adv2 = model(im, targ2, make_adv=True, **kwargs)
        print("Attack success rate2: %.4f" % (ch.sum(targ2 == ch.argmax(model_prediction2, dim=1)).item() / len(im)))

        model_prediction3, im_adv3 = model(im, targ3, make_adv=True, **kwargs)
        print("Attack success rate3: %.4f" % (ch.sum(targ3 == ch.argmax(model_prediction3, dim=1)).item() / len(im)))

        model_prediction4, im_adv4 = model(im, targ4, make_adv=True, **kwargs)
        print("Attack success rate4: %.4f" % (ch.sum(targ4 == ch.argmax(model_prediction4, dim=1)).item() / len(im)))

        # mixup and horizontal concat
        # bs * 3 * height * width
        lambd = opt.lambd
        alpha = opt.alpha
        im_adv = lambd * im_adv + (1 - lambd) * im_adv2
        im_adv3 = lambd * im_adv3 + (1 - lambd) * im_adv4
        dim_W = int(alpha * im.size()[3])
        im_adv_left = im_adv[:, :, :, :dim_W]
        im_adv_right = im_adv3[:, :, :, dim_W:]
        im_adv = ch.cat((im_adv_left, im_adv_right), dim=3)

        train_image.append(im_adv.cpu())
        train_label.append(label.cpu())
        time_ep = time.time() - time_ep
        print("Time:%.4f" % (time_ep))

    ch.save(train_image, os.path.join(opt.enc_data, 'train_image_mixandcat'))
    ch.save(train_label, os.path.join(opt.enc_data, 'train_label_mixandcat'))
    print("Save train set finished")


def generate_test_data_mixandcat(test_loader, model, kwargs, opt):

    test_image = []
    test_label = []
    correct = 0
    for i, (im, label) in enumerate(test_loader, 0):
        time_ep = time.time()
        print("process testset batch %d" % (i))
        im = im.cuda()
        predict, _ = model(im)
        label_pred = ch.argmax(predict, dim=1)
        targ = map_label_target(label_pred).cuda()
        targ2 = map_label_target2(label_pred).cuda()
        targ3 = map_label_target3(label_pred).cuda()
        targ4 = map_label_target4(label_pred).cuda()
        correct += ch.sum(label == label_pred.cpu()).item()

        model_prediction, im_adv = model(im, targ, make_adv=True, **kwargs)
        print("Attack success rate1: %.4f" % (ch.sum(targ == ch.argmax(model_prediction, dim=1)).item() / len(im)))

        model_prediction2, im_adv2 = model(im, targ2, make_adv=True, **kwargs)
        print("Attack success rate2: %.4f" % (ch.sum(targ2 == ch.argmax(model_prediction2, dim=1)).item() / len(im)))

        model_prediction3, im_adv3 = model(im, targ3, make_adv=True, **kwargs)
        print("Attack success rate3: %.4f" % (ch.sum(targ3 == ch.argmax(model_prediction3, dim=1)).item() / len(im)))

        model_prediction4, im_adv4 = model(im, targ4, make_adv=True, **kwargs)
        print("Attack success rate4: %.4f" % (ch.sum(targ4 == ch.argmax(model_prediction4, dim=1)).item() / len(im)))

        # mixup and horizontal concat
        # bs * 3 * height * width
        lambd = opt.lambd
        alpha = opt.alpha
        im_adv = lambd * im_adv + (1 - lambd) * im_adv2
        im_adv3 = lambd * im_adv3 + (1 - lambd) * im_adv4
        dim_W = int(alpha * im.size()[3])
        im_adv_left = im_adv[:, :, :, :dim_W]
        im_adv_right = im_adv3[:, :, :, dim_W:]
        im_adv = ch.cat((im_adv_left, im_adv_right), dim=3)

        test_image.append(im_adv.cpu())
        test_label.append(label.cpu())
        time_ep = time.time() - time_ep
        print("Time:%.4f" % (time_ep))

    ch.save(test_image, os.path.join(opt.enc_data, 'test_image_mixandcat'))
    ch.save(test_label, os.path.join(opt.enc_data, 'test_label_mixandcat'))
    print("Save test set finished")
    print("Orig test set acc:%.4f" % (correct / 10000))


def map_label_target(label):

    batch_size = len(label)
    target_map = [8, 3, 1, 0, 2, 4, 9, 6, 7, 5]
    targ_list = [target_map[label[i]] for i in range(batch_size)]
    targ = ch.tensor(targ_list)

    return targ


def map_label_target2(label):
    batch_size = len(label)
    target_map = [4, 2, 3, 5, 7, 1, 8, 0, 6, 9]
    targ_list = [target_map[label[i]] for i in range(batch_size)]
    targ = ch.tensor(targ_list)

    return targ


def map_label_target3(label):

    batch_size = len(label)
    target_map = [6, 2, 9, 1, 0, 7, 5, 3, 4, 8]
    targ_list = [target_map[label[i]] for i in range(batch_size)]
    targ = ch.tensor(targ_list)

    return targ


def map_label_target4(label):

    batch_size = len(label)
    target_map = [3, 9, 6, 1, 5, 8, 4, 7, 0, 2]
    targ_list = [target_map[label[i]] for i in range(batch_size)]
    targ = ch.tensor(targ_list)

    return targ


if __name__ == '__main__':
    main()

