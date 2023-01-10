import tensorflow as tf
import os.path
import numpy as np
from nets import TrajectoryCNN
from data_provider import datasets_factory_joints_h36m as datasets_factory
from utils import metrics
from utils import recoverh36m_3d
from utils import optimizer
import time
import scipy.io as io
import os, shutil
import pdb

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('is_training', 'True', 'train or not.')
# data path
tf.app.flags.DEFINE_string('dataset_name', 'skeleton',
                           'The name of dataset.')
tf.app.flags.DEFINE_string('train_data_paths',
                           'data/moving-mnist-example/moving-mnist-train.npz',
                           'train data paths.')
tf.app.flags.DEFINE_string('valid_data_paths',
                           'data/moving-mnist-example/moving-mnist-valid.npz',
                           'validation data paths.')
tf.app.flags.DEFINE_string('test_data_paths',
                           'data/moving-mnist-example/moving-mnist-valid.npz',
                           'test data paths.')
tf.app.flags.DEFINE_string('real_test_file', '',
                           'test data paths.')
tf.app.flags.DEFINE_string('save_dir', 'checkpoints/mnist_predcnn',
                           'dir to store trained net.')
tf.app.flags.DEFINE_string('gen_dir', 'results/mnist_predcnn',
                           'path to save generate results')
tf.app.flags.DEFINE_string('bak_dir', 'results/mnist_predcnn/bak',
                           'dir to backup result.')
# model parameter
tf.app.flags.DEFINE_string('pretrained_model', 'checkpoints/h36m/v1/model.ckpt-769500',
                           'file of a pretrained model to initialize from.')
tf.app.flags.DEFINE_integer('input_length', 10,
                            'encoder hidden states.')
tf.app.flags.DEFINE_integer('seq_length', 35,
                            'total input and output length.')
"""
------------------------------------改动1-----------------------------------------------------------------------
将'joints_number', 22改为14
改此处是因为第78行会用到
"""
tf.app.flags.DEFINE_integer('joints_number', 14,  # 17个点去除掉静态的3个点
                            'the number of joints of a pose')
tf.app.flags.DEFINE_integer('joint_dims', 3,
                            'one joints dims.')

tf.app.flags.DEFINE_integer('stacklength', 8,
                            'stack trajblock number.')
# tf.app.flags.DEFINE_integer('numhidden', '100,100,100,100,100',
#                            'trajblock filter number.')
tf.app.flags.DEFINE_integer('filter_size', 3,
                            'filter size.')

# opt
tf.app.flags.DEFINE_float('lr', 0.0001,
                          'base learning rate.')
tf.app.flags.DEFINE_integer('batch_size', 8,
                            'batch size for training.')
tf.app.flags.DEFINE_integer('max_iterations', 100000,
                            'max num of steps.')
tf.app.flags.DEFINE_integer('display_interval', 1,
                            'number of iters showing training loss.')
tf.app.flags.DEFINE_integer('test_interval', 20,
                            'number of iters for test.')
tf.app.flags.DEFINE_integer('snapshot_interval', 10000,
                            'number of iters saving models.')
tf.app.flags.DEFINE_integer('num_save_samples', 100000,
                            'number of sequences to be saved.')
tf.app.flags.DEFINE_integer('n_gpu', 4,
                            'how many GPUs to distribute the training across.')
num_hidden = [64, 64, 64, 64, 64]
print('!!! TrajectoryCNN:', num_hidden)


class Model(object):
    def __init__(self):
        if FLAGS.pretrained_model:
            print('!!!')
        else:
            print('???')
        # inputs
        self.x = [tf.placeholder(tf.float32, [FLAGS.batch_size,
                                              FLAGS.seq_length + FLAGS.seq_length - FLAGS.input_length,
                                              FLAGS.joints_number,
                                              FLAGS.joint_dims])
                  for i in range(FLAGS.n_gpu)]
        grads = []
        loss_train = []
        self.pred_seq = []
        self.tf_lr = tf.placeholder(tf.float32, shape=[])
        self.keep_prob = tf.placeholder(tf.float32)
        self.params = dict()

        for i in range(FLAGS.n_gpu):
            with tf.device('/gpu:%d' % i):
                with tf.variable_scope(tf.get_variable_scope(),
                                       reuse=True if i > 0 else None):
                    # define a model
                    output_list = TrajectoryCNN.TrajectoryCNN(
                        self.x[i],  # 传入的是一个tf.placeholder
                        self.keep_prob,
                        FLAGS.seq_length,
                        FLAGS.input_length,
                        FLAGS.stacklength,
                        num_hidden,
                        FLAGS.filter_size)

                    gen_ims = output_list[0]
                    loss = output_list[1]
                    pred_ims = gen_ims[:, FLAGS.input_length - FLAGS.seq_length:]
                    loss_train.append(loss)
                    # gradients
                    all_params = tf.trainable_variables()
                    grads.append(tf.gradients(loss, all_params))
                    self.pred_seq.append(pred_ims)

        if FLAGS.n_gpu == 1:
            self.train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)
        else:
            # add losses and gradients together and get training updates
            with tf.device('/gpu:0'):
                for i in range(1, FLAGS.n_gpu):
                    loss_train[0] += loss_train[i]
                    for j in range(len(grads[0])):
                        grads[0][j] += grads[i][j]
            # keep track of moving average
            ema = tf.train.ExponentialMovingAverage(decay=0.9995)
            maintain_averages_op = tf.group(ema.apply(all_params))
            self.train_op = tf.group(optimizer.adam_updates(
                all_params, grads[0], lr=self.tf_lr, mom1=0.95, mom2=0.9995),
                maintain_averages_op)

        self.loss_train = loss_train[0] / FLAGS.n_gpu

        # session
        variables = tf.global_variables()
        self.saver = tf.train.Saver(variables)
        init = tf.global_variables_initializer()
        configProt = tf.ConfigProto()
        configProt.gpu_options.allow_growth = True
        configProt.allow_soft_placement = True
        self.sess = tf.Session(config=configProt)
        self.sess.run(init)
        if FLAGS.pretrained_model:
            print('pretrain model: ', FLAGS.pretrained_model)
            self.saver.restore(self.sess, FLAGS.pretrained_model)

    def train(self, inputs, lr, keep_prob):
        feed_dict = {self.x[i]: inputs[i] for i in range(FLAGS.n_gpu)}
        feed_dict.update({self.tf_lr: lr})
        feed_dict.update({self.keep_prob: keep_prob})
        loss, _ = self.sess.run((self.loss_train, self.train_op), feed_dict)
        return loss

    def test(self, inputs, keep_prob):
        feed_dict = {self.x[i]: inputs[i] for i in range(FLAGS.n_gpu)}
        feed_dict.update({self.keep_prob: keep_prob})
        gen_ims = self.sess.run(self.pred_seq, feed_dict)
        return gen_ims

    def save(self, itr):
        checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
        self.saver.save(self.sess, checkpoint_path, global_step=itr)
        print('saved to ' + FLAGS.save_dir)


def tst(model):
    print('test...')
    res_path = os.path.join(FLAGS.gen_dir, 'test')
    if not tf.gfile.Exists(res_path):
        os.mkdir(res_path)
    test_time = 0
    print('loading inputs from', FLAGS.real_test_file)
    data = np.load(FLAGS.real_test_file)  # (338, 17, 3)

    steps = 1
    n = (len(data) - FLAGS.input_length) // steps + 1
    all_input = np.zeros((len(data), 20, 32, 3))
    trans_video_pose = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 6,
        5: 7,
        6: 8,
        7: 12,
        8: 13,
        9: 14,
        10: 15,
        11: 17,
        12: 18,
        13: 19,
        14: 25,
        15: 26,
        16: 27
    }
    trans_trajectory_cnn = {
        0: 30, 1: 29, 2: 27, 3: 26, 4: 25, 5: 17, 6: 18, 7: 19, 8: 21, 9: 22, 10: 15, 11: 14, 12: 13, 13: 12, 14: 7,
        15: 8,
        16: 9, 17: 10, 18: 2, 19: 3, 20: 4, 21: 5,
        # 重复点
        22: 16, 23: 20, 24: 23, 25: 24, 26: 28, 27: 31,
        # 不动点
        28: 0, 29: 1, 30: 6, 31: 11
    }
    new_trans_video_pose = dict(zip(trans_video_pose.values(), trans_video_pose.keys()))
    new_trans_trajectory_cnn = dict(zip(trans_trajectory_cnn.values(), trans_trajectory_cnn.keys()))
    index_list_tem = []
    index_list_data = np.arange(17)
    # 换节点顺序
    for i in range(17):
        index_list_tem.append(new_trans_trajectory_cnn[trans_video_pose[i]])
    all_input[:, 0, index_list_tem] = data[:, index_list_data]
    # 换维度
    for j in range(n):
        for i in range(0, FLAGS.input_length):
            all_input[j, i] = all_input[j * steps + i, 0]
    all_input = np.delete(all_input, range(n + 1, len(data)), axis=0)

    img_gen = np.ndarray((0, FLAGS.seq_length - FLAGS.input_length, FLAGS.joints_number, 3))
    for i in range(int(len(all_input) / FLAGS.batch_size)):
        start_time1 = time.time()
        tem = all_input[i * FLAGS.batch_size:i * FLAGS.batch_size + FLAGS.batch_size]
        tem = np.repeat(tem, FLAGS.n_gpu, axis=0)
        test_ims = tem[:, 0:FLAGS.seq_length, :, :]
        test_ims = test_ims[:, :, 0:22, :]
        test_ims = np.delete(test_ims, (0, 1, 8, 9, 16, 17, 20, 21), axis=2)
        test_dat = test_ims[:, 0:FLAGS.input_length, :, :]
        tem = test_dat[:, FLAGS.input_length - 1]
        tem = np.expand_dims(tem, axis=1)
        tem = np.repeat(tem, FLAGS.seq_length - FLAGS.input_length, axis=1)
        test_dat1 = np.concatenate((test_dat, tem), axis=1)
        test_dat2 = test_ims[:, FLAGS.input_length:]
        test_dat = np.concatenate((test_dat1, test_dat2), axis=1)
        test_dat = np.split(test_dat, FLAGS.n_gpu)
        test_gen = model.test(test_dat, 1)
        end_time1 = time.time()
        t1 = end_time1 - start_time1
        test_time += t1
        # concat outputs of different gpus along batch
        test_gen = np.concatenate(test_gen)
        img_gen = np.concatenate((img_gen, test_gen), axis=0)

    print(f'test time: {test_time}')

    # 换维度
    save_data_1 = np.zeros(((n - n % FLAGS.batch_size - 1) * steps + FLAGS.input_length, FLAGS.joints_number, 3))
    for ik in range(n - n % FLAGS.batch_size):
        for step in range(steps):
            save_data_1[ik * steps + step] = img_gen[
                ik * FLAGS.n_gpu, (FLAGS.seq_length - FLAGS.input_length - steps) // 2 + step]
    static_joints = np.array(
        [[[-1.32948593e+02, 0.00000000e+00, 0.00000000e+00],
         [1.32948822e+02, 0.00000000e+00, 0.00000000e+00]]]
    )

    # 换结点顺序
    index_before = np.array([2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 18, 19])
    index_after = []
    for i in index_before:
        index_after.append(new_trans_video_pose[trans_trajectory_cnn[i]])
    save_data_2 = np.zeros((len(save_data_1), 17, 3))
    save_data_2[:, index_after] = save_data_1[:, np.arange(14)]

    # save_data_2[:, [1, 4]] = np.repeat(static_joints, len(save_data_2), axis=0)
    save_data_2[:, [1, 4]] = data[:save_data_2.shape[0], [1, 4]]
    save_data_2 = np.insert(save_data_2, np.zeros((10, ), dtype=np.intp), values=np.zeros(save_data_2.shape[1:]), axis=0)

    # save prediction examples
    save_data_3 = save_data_2.reshape(-1)
    save_data_3 = save_data_3 / (np.max(save_data_3) - np.min(save_data_3)) * 2
    save_data_3 = save_data_3.reshape((-1, 17, 3))
    path = res_path
    if not tf.gfile.Exists(path):
        os.mkdir(path)
    np.save(os.path.join(path, 'all_input.npy'), all_input)
    np.save(path, save_data_3)
    np.save(os.path.join(path, 'img_gen.npy'), img_gen)
    print('save test done!')


def main(argv=None):
    if not tf.gfile.Exists(FLAGS.save_dir):
        tf.gfile.MakeDirs(FLAGS.save_dir)
    if not tf.gfile.Exists(FLAGS.gen_dir):
        tf.gfile.MakeDirs(FLAGS.gen_dir)

    print('start training !', time.strftime('%Y-%m-%d %H:%M:%S\n\n\n', time.localtime(time.time())))

    print('Initializing models')
    model = Model()
    lr = FLAGS.lr
    train_time = 0
    test_time_all = 0
    folder = 1
    path_bak = FLAGS.bak_dir

    min_err = 100000.0
    err_list = []
    Keep_prob = 0.75

    if FLAGS.is_training == 'False':
        tst(model)

    if FLAGS.is_training == 'True':
        # load data
        train_input_handle, test_input_handle = datasets_factory.data_provider(
            FLAGS.dataset_name, FLAGS.train_data_paths, FLAGS.valid_data_paths,
            FLAGS.batch_size * FLAGS.n_gpu, FLAGS.joints_number, FLAGS.input_length, FLAGS.seq_length, is_training=True)

        for itr in range(1, FLAGS.max_iterations + 1):
            if train_input_handle.no_batch_left():  # train_input_handle是一个joints_h36m.py下类的对象
                train_input_handle.begin(do_shuffle=True)
            '''
            if itr % 20000 == 0:
                lr = lr* 0.95
            '''
            start_time = time.time()
            ims = train_input_handle.get_batch()
            ims = ims[:, :, 0:22, :]  # 取了前22个点
            """
            ------------------------------------改动2-----------------------------------------------------------------------
            添加了一行
            ims = np.delete(ims, (0, 1, 8, 9, 16, 17, 20, 21), axis=2)
            此处是train数据集的处理
            """
            ims = np.delete(ims, (0, 1, 8, 9, 16, 17, 20, 21), axis=2)
            pretrain_iter = 0
            if itr < pretrain_iter:
                inputs1 = ims
            else:
                inputs1 = ims[:, 0:FLAGS.input_length, :, :]
                tem = ims[:, FLAGS.input_length - 1]  # tem的维度为(batch中有多少组数据,关节数,3),这个操作是把最后一帧取出
                tem = np.expand_dims(tem, axis=1)  # 将第二个维度补出来(数组组数,帧数,关节数,3)
                tem = np.repeat(tem, FLAGS.seq_length - FLAGS.input_length, axis=1)  # 在帧数的维度上扩展成预测的帧数
                inputs1 = np.concatenate((inputs1, tem), axis=1)  # 相当于把输入的最后一帧一直重复
            # pdb.set_trace()
            inputs2 = ims[:, FLAGS.input_length:]  # 只留下预测的帧
            inputs = np.concatenate((inputs1, inputs2), axis=1)  # 连到一起
            ims_list = np.split(inputs, FLAGS.n_gpu)
            cost = model.train(ims_list, lr, Keep_prob)
            # inverse the input sequence
            imv1 = ims[:, ::-1]
            if itr >= pretrain_iter:
                imv_rev1 = imv1[:, 0:FLAGS.input_length, :, :]
                # pdb.set_trace()
                tem = imv1[:, FLAGS.input_length - 1]
                tem = np.expand_dims(tem, axis=1)
                tem = np.repeat(tem, FLAGS.seq_length - FLAGS.input_length, axis=1)
                # pdb.set_trace()
                imv_rev1 = np.concatenate((imv_rev1, tem), axis=1)
            else:
                imv_rev1 = imv1
            imv_rev2 = imv1[:, FLAGS.input_length:]
            ims_rev1 = np.concatenate((imv_rev1, imv_rev2), axis=1)
            ims_rev1 = np.split(ims_rev1, FLAGS.n_gpu)
            cost += model.train(ims_rev1, lr, Keep_prob)
            cost = cost / 2

            end_time = time.time()
            t = end_time - start_time
            train_time += t

            if itr % FLAGS.display_interval == 0:
                print('itr: ' + str(itr) + ' lr: ' + str(lr) + ' training loss: ' + str(cost))

            if itr % FLAGS.test_interval == 0:
                print('train time:' + str(train_time))
                print('test...')
                str1 = 'walking eating smoking discussion directions greeting phoning posing purchases sitting ' \
                       'sittingdown takingphoto waiting walkingdog walkingtogether'
                str1 = str1.split(' ')
                res_path = os.path.join(FLAGS.gen_dir, str(itr))
                if not tf.gfile.Exists(res_path):
                    os.mkdir(res_path)
                avg_mse = 0
                batch_id = 0
                test_time = 0
                joint_mse = np.zeros((25, 32))
                joint_mae = np.zeros((25, 32))
                mpjpe = np.zeros([1, FLAGS.seq_length - FLAGS.input_length])
                mpjpe_l = np.zeros([1, FLAGS.seq_length - FLAGS.input_length])
                img_mse, ssim, psnr, fmae, sharp = [], [], [], [], []
                for i in range(FLAGS.seq_length - FLAGS.input_length):
                    img_mse.append(0)
                    fmae.append(0)
                f = 0
                for s in str1:
                    start_time1 = time.time()
                    batch_id = batch_id + 1
                    mpjpe1 = np.zeros([1, FLAGS.seq_length - FLAGS.input_length])
                    tem = np.load(FLAGS.test_data_paths + '/' + s + '.npy')
                    tem = np.repeat(tem, (FLAGS.batch_size * FLAGS.n_gpu) / 8, axis=0)
                    test_ims = tem[:, 0:FLAGS.seq_length, :, :]
                    test_ims1 = test_ims
                    test_ims = test_ims[:, :, 0:22, :]
                    """
                    ------------------------------------改动3-----------------------------------------------------------------------
                    添加了一行test_ims=np.delete(test_ims, (0, 1, 8, 9, 16, 17, 20, 21), axis=2)
                    此处是test数据集的处理
                    """
                    test_ims = np.delete(test_ims, (0, 1, 8, 9, 16, 17, 20, 21), axis=2)
                    test_dat = test_ims[:, 0:FLAGS.input_length, :, :]
                    tem = test_dat[:, FLAGS.input_length - 1]
                    tem = np.expand_dims(tem, axis=1)
                    tem = np.repeat(tem, FLAGS.seq_length - FLAGS.input_length, axis=1)
                    test_dat1 = np.concatenate((test_dat, tem), axis=1)
                    test_dat2 = test_ims[:, FLAGS.input_length:]
                    test_dat = np.concatenate((test_dat1, test_dat2), axis=1)
                    test_dat = np.split(test_dat, FLAGS.n_gpu)
                    img_gen = model.test(test_dat, 1)
                    end_time1 = time.time()
                    t1 = end_time1 - start_time1
                    test_time += t1
                    # concat outputs of different gpus along batch
                    img_gen = np.concatenate(img_gen)
                    gt_frm = test_ims1[:, FLAGS.input_length:]
                    img_gen = recoverh36m_3d.recoverh36m_3d(gt_frm, img_gen)  # 将关键点恢复成gt的样子，用于计算损失

                    # mpjpe1=np.zeros([1,FLAGS.seq_length - FLAGS.input_length])
                    # MSE per frame
                    for i in range(FLAGS.seq_length - FLAGS.input_length):
                        x = gt_frm[:, i, :, ]
                        gx = img_gen[:, i, :, ]
                        fmae[i] += metrics.batch_mae_frame_float(gx, x)
                        mse = np.square(x - gx).sum()
                        for j in range(FLAGS.batch_size * FLAGS.n_gpu):
                            tem1 = 0
                            for k in range(32):
                                tem1 += np.sqrt(np.square(x[j, k] - gx[j, k]).sum())
                            mpjpe1[0, i] += tem1 / 32

                        img_mse[i] += mse
                        avg_mse += mse
                        real_frm = x
                        pred_frm = gx
                        for j in range(32):
                            xi = x[:, j]
                            gxi = gx[:, j]
                            joint_mse[i, j] += np.square(xi - gxi).sum()
                            joint_mae[i, j] += metrics.batch_mae_frame_float1(gxi, xi)
                    # save prediction examples
                    path = os.path.join(res_path, s)
                    if not tf.gfile.Exists(path):
                        os.mkdir(path)
                    for ik in range(8):
                        spath = os.path.join(path, str(ik))
                        if not tf.gfile.Exists(spath):
                            os.mkdir(spath)
                        for i in range(FLAGS.seq_length):
                            name = 'gt' + str(i + 1) + '.mat'
                            file_name = os.path.join(spath, name)
                            img_gt = test_ims1[ik * FLAGS.n_gpu*FLAGS.batch_size//8, i, :, :]
                            io.savemat(file_name, {'joint': img_gt})
                        for i in range(FLAGS.seq_length - FLAGS.input_length):
                            name = 'pd' + str(i + 1 + FLAGS.input_length) + '.mat'
                            file_name = os.path.join(spath, name)
                            img_pd = img_gen[ik * FLAGS.n_gpu*FLAGS.batch_size//8, i, :, :]
                            io.savemat(file_name, {'joint': img_pd})
                    mpjpe1 = mpjpe1 / (FLAGS.batch_size * FLAGS.n_gpu)
                    print('current action mpjpe: ', s)
                    for i in mpjpe1[0]:
                        print(i)
                    mpjpe += mpjpe1
                    if f <= 3:
                        print('four actions', s)
                        mpjpe_l += mpjpe1
                    f = f + 1
                test_time_all += test_time
                joint_mae = np.asarray(joint_mae, dtype=np.float32) / batch_id
                joint_mse = np.asarray(joint_mse, dtype=np.float32) / (batch_id * FLAGS.batch_size * FLAGS.n_gpu)
                avg_mse = avg_mse / (batch_id * FLAGS.batch_size * FLAGS.n_gpu)
                print('mse per seq: ' + str(avg_mse))
                # for i in range(FLAGS.seq_length - FLAGS.input_length):
                #	print(img_mse[i] / (batch_id * FLAGS.batch_size * FLAGS.n_gpu))
                mpjpe = mpjpe / (batch_id)
                err_list.append(np.mean(mpjpe))
                print('mean per joints position error: ' + str(np.mean(mpjpe)))
                for i in range(FLAGS.seq_length - FLAGS.input_length):
                    print(mpjpe[0, i])
                mpjpe_l = mpjpe_l / 4
                print('mean mpjpe for four actions: ' + str(np.mean(mpjpe_l)))
                for i in range(FLAGS.seq_length - FLAGS.input_length):
                    print(mpjpe_l[0, i])
                fmae = np.asarray(fmae, dtype=np.float32) / batch_id
                print('fmae per frame: ' + str(np.mean(fmae)))
                # for i in range(FLAGS.seq_length - FLAGS.input_length):
                #	print(fmae[i])
                print('current test time:' + str(test_time))
                print('all test time: ' + str(test_time_all))
                filename = os.path.join(res_path, 'test_result')
                io.savemat(filename, {'joint_mse': joint_mse, 'joint_mae': joint_mae, 'mpjpe': mpjpe})

            if itr % FLAGS.snapshot_interval == 0 and min(err_list) < min_err:
                model.save(itr)
                min_err = min(err_list)
                print('model saving done! ', time.strftime('%Y-%m-%d %H:%M:%S\n\n\n', time.localtime(time.time())))

            if itr % FLAGS.snapshot_interval == 0:
                print('current minimize error is: ', min_err)

            train_input_handle.next()


if __name__ == '__main__':
    tf.app.run()
