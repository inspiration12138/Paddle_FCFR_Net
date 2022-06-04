# load data
from reprod_log import ReprodDiffHelper

if __name__ == '__main__':

    diff_helper_train = ReprodDiffHelper()
    torch_train_info = diff_helper_train.load_info("../result/data_train_ref.npy")
    paddle_train_info = diff_helper_train.load_info("../result/data_train_paddle.npy")

    diff_helper_val = ReprodDiffHelper()
    torch_val_info = diff_helper_val.load_info("../result/data_val_ref.npy")
    paddle_val_info = diff_helper_val.load_info("../result/data_val_paddle.npy")


    # compare result and produce log
    print('train')
    diff_helper_train.compare_info(torch_train_info, paddle_train_info)
    diff_helper_train.report(path="../result/log/data_train_diff.log")

    print('val')
    diff_helper_val.compare_info(torch_val_info, paddle_val_info)
    diff_helper_val.report(path="../result/log/data_val_diff.log")