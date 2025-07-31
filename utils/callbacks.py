import os
import matplotlib
import torch

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import scipy.signal


class LogWriter():
    def __init__(self, log_dir, model, input_shape, device):
        self.log_dir = log_dir
        self.lr = []
        self.train_loss = []
        self.train_mae = []
        self.train_f1 = []
        self.val_loss = []
        self.val_mae = []
        self.val_f1 = []
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        try:
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1]).to(device)
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_train_value(self, epoch, train_loss, train_mae, train_f1, lr):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.train_loss.append(train_loss)
        self.train_mae.append(train_mae)
        self.train_f1.append(train_f1)
        self.lr.append(lr)

        with open(os.path.join(self.log_dir, "epoch_log_train.txt"), 'a') as f:
            f.write("{} {} {} {} {}\n".format(str(epoch), str(train_loss),
                                              str(train_mae), str(train_f1),
                                              str(lr)))
        self.writer.add_scalar('Lr', lr, epoch)
        self.writer.add_scalar('Train/Train_Loss', train_loss, epoch)
        self.writer.add_scalar('Train/Train_MAE', train_mae, epoch)
        self.writer.add_scalar('Train/Train_F1', train_f1, epoch)
        # self.plot_loss()
        # self.plot_mae()
        # self.plot_f1()

    def append_val_value(self, epoch, val_loss, val_mae, val_f1):
        self.val_loss.append(val_loss)
        self.val_mae.append(val_mae)
        self.val_f1.append(val_f1)

        with open(os.path.join(self.log_dir, "epoch_log_val.txt"), 'a') as f:
            f.write("{} {} {} {}\n".format(str(epoch), str(val_loss), str(val_mae), str(val_f1)))
        self.writer.add_scalar('Val/Val_Loss', val_loss, epoch)
        self.writer.add_scalar('Val/Val_MAE', val_mae, epoch)
        self.writer.add_scalar('Val/Val_F1', val_f1, epoch)
        self.plot_loss()
        self.plot_mae()
        self.plot_f1()

    def plot_loss(self):
        train_iters = range(len(self.train_loss))
        val_iters = range(len(self.val_loss))
        plt.figure()
        plt.plot(train_iters, self.train_loss, 'red', linewidth=2, label='train loss')
        plt.plot(val_iters, self.val_loss, 'blue', linewidth=2, label='val loss')

        try:
            if len(self.train_loss) < 25:
                num = 5
            else:
                num = 15

            plt.plot(train_iters, scipy.signal.savgol_filter(self.train_loss, num, 3), 'green', linestyle='--',
                     linewidth=2,
                     label='smooth train loss')
            plt.plot(val_iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--',
                     linewidth=2,
                     label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        plt.cla()
        plt.close("all")

    def plot_mae(self):
        train_iters = range(len(self.train_mae))
        val_iters = range(len(self.val_mae))
        plt.figure()
        plt.plot(train_iters, self.train_mae, 'red', linewidth=2, label='train MAE')
        plt.plot(val_iters, self.val_mae, 'blue', linewidth=2, label='val MAE')

        try:
            if len(self.train_mae) < 25:
                num = 5
            else:
                num = 15

            plt.plot(train_iters, scipy.signal.savgol_filter(self.train_mae, num, 3), 'green', linestyle='--',
                     linewidth=2,
                     label='smooth train MAE')
            plt.plot(val_iters, scipy.signal.savgol_filter(self.val_mae, num, 3), '#8B4513', linestyle='--',
                     linewidth=2,
                     label='smooth val MAE')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_MAE.png"))
        plt.cla()
        plt.close("all")

    def plot_f1(self):
        train_iters = range(len(self.train_f1))
        val_iters = range(len(self.val_f1))
        plt.figure()
        plt.plot(train_iters, self.train_f1, 'red', linewidth=2, label='train F1')
        plt.plot(val_iters, self.val_f1, 'blue', linewidth=2, label='val F1')

        try:
            if len(self.train_f1) < 25:
                num = 5
            else:
                num = 15

            plt.plot(train_iters, scipy.signal.savgol_filter(self.train_f1, num, 3), 'green', linestyle='--',
                     linewidth=2,
                     label='smooth train F1')
            plt.plot(val_iters, scipy.signal.savgol_filter(self.val_f1, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='smooth val F1')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('F1')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_F1.png"))
        plt.cla()
        plt.close("all")
