from reprod_log import ReprodDiffHelper
import numpy as np

if __name__ == '__main__':
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("../result/losses_ref_no_eval.npy")
    paddle_info = diff_helper.load_info("../result/losses_paddle.npy")

    print(torch_info)
    print(paddle_info)

    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="../result/log/backward_diff.log")
